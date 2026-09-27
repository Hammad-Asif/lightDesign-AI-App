"""Offline contract tests; no external API, credentials, or Postgres required."""
import base64
import io
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.providers import roofline_v3 as v3
from app.providers.base import Detection
from app.routers import photos, regions
from app.storage import get_store


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def response(lines=None, **kwargs):
    data = {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"lines": lines if lines is not None else [{"type": "eave", "points": [0, 20, 90, 20]}]})}}],
        "id": "test", "usage": {"prompt_tokens": 123, "completion_tokens": 45, "cost": 0.01}}
    data.update(kwargs)
    return httpx.Response(200, json=data, request=httpx.Request("POST", v3.API))


def test_payload_matches_cli_and_keeps_original_pixel_frame(monkeypatch):
    post = Mock(return_value=response())
    monkeypatch.setattr(v3.httpx, "post", post)
    lines, meta = v3.detect(Image.new("RGB", (101, 37)), settings(openrouter_api_key="test-only"))
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "openai/gpt-6-sol"
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_completion_tokens"] == 8000
    assert payload["provider"] == {"require_parameters": True}
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "101 pixels wide and 37 pixels high" in payload["messages"][0]["content"]
    encoded = payload["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (101, 37)
    assert lines[0]["points"] == [0, 20, 90, 20]
    assert meta["cost_usd"] == 0.01 and meta["image_size"] == [101, 37]


@pytest.mark.parametrize("line", [
    {"type": "eave", "points": [True, 0, 10, 1]},
    {"type": "eave", "points": [0.0, 0, 10, 1]},
    {"type": "eave", "points": [0, 0, 100, 1]},
    {"type": "eave", "points": [-1, 0, 10, 1]},
    {"type": "eave", "points": [0, 0, 0, 0]},
    {"type": "roof", "points": [0, 0, 10, 1]},
    {"type": "eave", "points": [0, 0, 10]},
    {"type": "eave", "points": [0, 0, 10, 1], "extra": 1},
])
def test_rejects_invalid_geometry(line):
    with pytest.raises(ValueError):
        v3.validate({"lines": [line]}, (100, 40))


def test_missing_key_fails_before_network(monkeypatch):
    post = Mock()
    monkeypatch.setattr(v3.httpx, "post", post)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        v3.detect(Image.new("RGB", (100, 40)), settings())
    post.assert_not_called()


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_incomplete_response_is_not_accepted(monkeypatch, finish):
    raw = response().json()
    raw["choices"][0]["finish_reason"] = finish
    monkeypatch.setattr(v3.httpx, "post", Mock(return_value=response(**raw)))
    with pytest.raises(RuntimeError, match="incomplete"):
        v3.detect(Image.new("RGB", (100, 40)), settings(openrouter_api_key="test-only"))


def test_provider_failure_does_not_expose_body_or_retry(monkeypatch):
    reply = httpx.Response(401, text="sensitive provider detail", request=httpx.Request("POST", v3.API))
    post = Mock(return_value=reply)
    monkeypatch.setattr(v3.httpx, "post", post)
    with pytest.raises(RuntimeError, match="HTTP 401") as caught:
        v3.detect(Image.new("RGB", (100, 40)), settings(openrouter_api_key="test-only"))
    assert "sensitive" not in str(caught.value)
    assert post.call_count == 1


class MemoryStore:
    """Route test double. Production database persistence is covered by tests/smoke.py."""
    def __init__(self):
        self.items = []
        self.image = None
        self.row = None

    def create_photo(self, data, prompt, max_edge):
        self.image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        self.image.thumbnail((max_edge, max_edge))
        self.row = SimpleNamespace(id=uuid.uuid4(), prompt=prompt, width=self.image.width,
                                  height=self.image.height, mode=None, approved=False, interpreted=None)
        return self.row

    def photo(self, _): return self.row
    def load_image(self, _): return self.image
    def products(self): return []
    def update_photo(self, photo, **values):
        for name, value in values.items(): setattr(photo, name, value)
        return photo
    def replace_auto_regions(self, photo, fresh):
        self.items = [r for r in self.items if r.source == "manual"] + fresh
        for r in self.items:
            r.id = r.id or uuid.uuid4()
            r.role = r.role or "target"
    def refresh_hidden(self, _): return self.items
    def regions(self, _): return self.items
    def region(self, photo, region_id, **kwargs):
        return next(r for r in self.items if str(r.id) == region_id)
    def save(self): pass
    def write_mask(self, *args): return "/test-mask.png"


@pytest.fixture
def route_app(monkeypatch):
    store = MemoryStore()
    cfg = settings(roofline_detector="v3_sol", edge_extractor="partition", partition_engine="reference")
    monkeypatch.setattr(regions, "get_settings", lambda: cfg)
    monkeypatch.setattr(photos, "get_settings", lambda: cfg)
    monkeypatch.setattr(regions, "get_provider", lambda: Mock())
    app = FastAPI()
    app.include_router(photos.router)
    app.include_router(regions.router)
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app), store, cfg


def upload(client, size=(100, 40), prompt="roofline"):
    file = io.BytesIO()
    Image.new("RGB", size).save(file, "PNG")
    result = client.post("/photos", files={"file": ("roof.png", file.getvalue(), "image/png")}, data={"prompt": prompt})
    assert result.status_code == 200, result.text
    return result.json()["id"]


@pytest.mark.parametrize("size", [(100, 40), (801, 603), (900, 600), (1500, 2100), (1254, 1254)])
def test_upload_detect_edit_roundtrip(route_app, monkeypatch, size):
    client, store, _ = route_app
    detector = Mock(return_value=([{"type": "eave", "points": [0, 20, size[0]-1, 20]}], {"cost_usd": .01}))
    monkeypatch.setattr(v3, "detect", detector)
    sam = Mock(side_effect=AssertionError("SAM must not run for direct roof-only requests"))
    monkeypatch.setattr(regions, "get_provider", sam)
    photo_id = upload(client, size)
    result = client.post(f"/photos/{photo_id}/detect", json={"kinds": ["roof"]})
    assert result.status_code == 200, result.text
    line = result.json()[0]
    assert line["polyline"] == [[0, 20], [size[0]-1, 20]]
    assert line["edge"] == "eave" and line["visible"] and line["side"] is None
    assert detector.call_args.args[0].size == size
    changed = client.patch(f"/photos/{photo_id}/regions/{line['id']}", json={"polyline": [[1, 22], [90, 23]], "edge": "rake"})
    assert changed.status_code == 200
    assert changed.json()["edge"] == "rake" and changed.json()["polyline"][0] == [1, 22]
    assert client.get(f"/photos/{photo_id}").json()["interpreted"]["detection_run"]["cost_usd"] == .01
    sam.assert_not_called()


def test_flag_off_preserves_segmentation_route(route_app, monkeypatch):
    client, store, cfg = route_app
    cfg.roofline_detector = "sam"
    cfg.edge_extractor = "hybrid"
    cfg.detect_blockers = False
    direct = Mock(side_effect=AssertionError("V3 must not run"))
    monkeypatch.setattr(v3, "detect", direct)
    pipeline = Mock(return_value=[])
    monkeypatch.setattr(regions.detect, "run", pipeline)
    photo_id = upload(client)
    result = client.post(f"/photos/{photo_id}/detect", json={"kinds": ["roof"]})
    assert result.status_code == 200 and result.json() == []
    assert pipeline.call_args.args[2:] == (["roof"], False)
    direct.assert_not_called()


def test_mixed_request_only_segments_non_roof_kinds(route_app, monkeypatch):
    client, _, _ = route_app
    monkeypatch.setattr(v3, "detect", Mock(return_value=([], {})))
    detection = Detection("hedge", "hedge", np.ones((40, 100), bool), .9, (0, 0, 100, 40))
    pipeline = Mock(return_value=[detection])
    monkeypatch.setattr(regions.detect, "run", pipeline)
    photo_id = upload(client)
    result = client.post(f"/photos/{photo_id}/detect", json={"kinds": ["roof", "hedge"]})
    assert result.status_code == 200 and result.json()[0]["kind"] == "hedge"
    assert pipeline.call_args.args[2:] == (["hedge"], False)


def test_non_roof_request_never_calls_v3(route_app, monkeypatch):
    client, _, cfg = route_app
    cfg.edge_extractor = "hybrid"
    direct = Mock(side_effect=AssertionError("non-roof requests must not call V3"))
    monkeypatch.setattr(v3, "detect", direct)
    pipeline = Mock(return_value=[])
    monkeypatch.setattr(regions.detect, "run", pipeline)
    photo_id = upload(client)
    result = client.post(f"/photos/{photo_id}/detect", json={"kinds": ["hedge"]})
    assert result.status_code == 200
    assert pipeline.call_args.args[2] == ["hedge"]
    direct.assert_not_called()


def test_failed_detection_keeps_existing_regions(route_app, monkeypatch):
    client, store, _ = route_app
    photo_id = upload(client)
    sentinel = object()
    store.items = [sentinel]
    monkeypatch.setattr(v3, "detect", Mock(side_effect=RuntimeError("provider failed")))
    result = client.post(f"/photos/{photo_id}/detect", json={"kinds": ["roof"]})
    assert result.status_code == 503 and store.items == [sentinel]


def test_side_and_edge_filters_keep_unrequested_lines_for_review(route_app, monkeypatch):
    client, _, _ = route_app
    lines = [{"type": t, "points": [0, y, 90, y]} for t, y in (("eave", 10), ("ridge", 20))]
    monkeypatch.setattr(v3, "detect", Mock(return_value=(lines, {})))
    photo_id = upload(client, prompt="roofline")
    result = client.post(f"/photos/{photo_id}/detect", json={}).json()
    assert [r["visible"] for r in result] == [True, False]
    photo_id = upload(client, prompt="front roofline")
    result = client.post(f"/photos/{photo_id}/detect", json={}).json()
    assert all(not r["visible"] and r["side"] is None for r in result)
    assert client.get(f"/photos/{photo_id}").json()["interpreted"]["detection_run"]["side_review_required"]
