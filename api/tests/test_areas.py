"""M3 Day 4: palm trunk spirals, canopy and frond estimates, wreaths (services/placement.py)."""
import io
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.schemas import LightingSettings
from app.services import placement, scale
from test_scale import FT, Shot, facade_names, flush_eaves  # noqa: F401  (fixture)

MINI = SimpleNamespace(id=uuid.uuid4(), category="mini", bulb_diameter_in=0.2, spacing_in=4.0, spacing_options_in=[],
                       colours=["warm_white"], twinkle=False, drops_in=[], diameter_in=None, bulb_count=70,
                       strand_length_ft=24.0, density_per_10sqft=100.0, owner_id=None)
WREATH_48 = SimpleNamespace(id=uuid.uuid4(), category="wreath", bulb_diameter_in=None, spacing_in=None,
                            spacing_options_in=[], colours=["warm_white"], twinkle=False, drops_in=[], diameter_in=48.0,
                            bulb_count=150, strand_length_ft=None, owner_id=None, archived=False)
PER_INCH = 2.0   # px per inch where these masks stand


def settings(**extra):
    return LightingSettings(product_id=MINI.id, **extra)


def trunk_mask(width_in=12.0, height_in=180.0, size=(600, 800)):
    """A straight trunk of known size, standing at the bottom of the frame."""
    mask = np.zeros(size[::-1], bool)
    w, h = int(width_in * PER_INCH), int(height_in * PER_INCH)
    x0, bottom = 300 - w // 2, size[1] - 20
    mask[bottom - h:bottom, x0:x0 + w] = True
    return mask


def test_trunk_spiral_reads_the_trunk_at_the_local_scale_and_draws_only_the_near_half():
    mask = trunk_mask()
    placed = placement.spiral(mask, settings(wrap={"turn_spacing_in": 6.0}), MINI, "palm_trunk", PER_INCH, {})
    assert placed["trunk"]["diameter_in"] == pytest.approx(12.0, abs=0.5)
    assert placed["trunk"]["height_ft"] == pytest.approx(15.0, abs=0.1)
    turns = 180.0 / 6.0
    estimate_ft = turns * np.hypot(np.pi * 12.0, 6.0) / 12
    assert placed["strand_estimate_ft"] == pytest.approx(estimate_ft, rel=0.03)
    assert placed["estimated"] is True
    # About half of every turn is behind the trunk: counted, flagged, never drawn.
    assert placed["hidden"] == pytest.approx(placed["count"] / 2, rel=0.05)
    assert placed["count"] == int(placed["strand_ft"] * 12 // 4) + 1


def test_a_strand_length_the_salesperson_types_replaces_the_estimate():
    placed = placement.spiral(trunk_mask(), settings(), MINI, "palm_trunk", PER_INCH, {"strand_ft": 40})
    assert placed["strand_ft"] == 40 and placed["strand_set"] is True and placed["estimated"] is False
    assert placed["count"] == 40 * 12 // 4 + 1


def test_canopy_count_is_area_times_density_with_presets_and_an_override():
    mask = np.zeros((800, 600), bool)
    mask[100:100 + 120, 100:100 + 240] = True          # 60 x 120 in = 50 sq ft at 2 px per inch
    medium = placement.scatter(mask, settings(), MINI, "tree_canopy", PER_INCH, {}, "a")
    assert medium["area_sqft"] == pytest.approx(50.0, rel=0.01)
    assert medium["count"] == 500 and medium["estimated"] is True
    light = placement.scatter(mask, settings(), MINI, "tree_canopy", PER_INCH, {"density": "light"}, "a")
    heavy = placement.scatter(mask, settings(), MINI, "tree_canopy", PER_INCH, {"density": "heavy"}, "a")
    assert (light["count"], heavy["count"]) == (250, 1000)
    typed = placement.scatter(mask, settings(), MINI, "tree_canopy", PER_INCH, {"count": 320}, "a")
    assert typed["count"] == 320 and typed["estimated"] is False and typed["estimate"] == 500
    # Every scattered bulb is on the canopy, and the same ones come back every time.
    assert all(mask[int(b[1]), int(b[0])] for b in medium["bulbs"])
    assert medium["bulbs"] == placement.scatter(mask, settings(), MINI, "tree_canopy", PER_INCH, {}, "a")["bulbs"]


def test_a_wreath_is_a_ring_of_its_bulbs_at_its_real_size():
    placed = placement.wreath([200.0, 150.0], WREATH_48, 48, PER_INCH)
    assert placed["count"] == 150 and placed["ring"][2] == pytest.approx(96.0)
    radii = [np.hypot(b[0] - 200, b[1] - 150) for b in placed["bulbs"]]
    assert radii == pytest.approx([placement.WREATH_RING * 96.0] * 150, abs=0.1)


def test_local_scale_comes_from_the_wall_behind_the_spot_not_the_nearest_plane(flush_eaves):
    shot = Shot()
    cam, front, side = facade_names(shot)
    lines = [shot.line("front", (0, 10, 0), (40, 10, 0), "eave", front),
             shot.line("side", (40, 10, 0), (40, 10, 30), "eave", side)]
    solved = scale.solve(cam, lines, {"facade": front, "points": lines[0].points, "length_in": 40 * FT})
    pixels = {line.id: line.points for line in lines}
    at_front = shot.pixel((20, 3, 0))                 # low on the front wall's face
    per_inch, facade = placement.wall_scale(cam, solved, pixels, at_front)
    assert facade == front
    assert per_inch == pytest.approx(cam.focal / (shot.cam((20, 3, 0))[2] * FT), rel=0.01)
    assert placement.wall_scale(cam, solved, pixels, shot.pixel((40, 3, 20)))[1] == side


@pytest.fixture
def client(monkeypatch):
    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.config import Settings
    from app.routers import photos
    from app.storage import get_store
    from test_roofline_v3 import MemoryStore

    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("no network")))
    cfg = Settings(_env_file=None, segmentation_provider="mock", instruction_parser="rules")
    monkeypatch.setattr(photos, "get_settings", lambda: cfg)

    class Store(MemoryStore):
        masks: dict = {}

        def add_region(self, photo, region):
            self.items.append(region)
            return region

        def write_mask(self, photo_id, region_id, mask):
            self.masks[str(region_id)] = mask
            return f"/m/{region_id}.png"

        def read_mask(self, photo_id, region_id):
            return self.masks.get(str(region_id))

        def design(self, photo):
            return None

        def products(self, *a, **k):
            return [] if k.get("archived") else [WREATH_48]

    store = Store()
    api = FastAPI()
    api.include_router(photos.router)
    api.dependency_overrides[get_store] = lambda: store
    test_client = TestClient(api)
    buffer = io.BytesIO()
    Image.new("RGB", (1600, 1200)).save(buffer, "PNG")
    photo_id = test_client.post("/photos", files={"file": ("h.png", buffer.getvalue(), "image/png")},
                                data={"prompt": ""}).json()["id"]
    shot = Shot()
    cam, front, _ = facade_names(shot)
    store.row.interpreted = {"camera": cam.to_dict()}
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    from app.models import Region as RegionRow
    store.items = [RegionRow(id=uuid.uuid4(), kind="roof", geometry="line", edge="eave", side=front, role="target",
                             source="auto", label="Eave", color="#39ff14", score=0.9, visible=True, mask_url=None,
                             polyline=[[round(x), round(y)] for x, y in eave.points], hidden=None)]
    return test_client, store, photo_id, shot, eave


def test_a_wreath_needs_a_measurement_then_is_drawn_at_its_real_size_and_resized(client):
    test_client, store, photo_id, shot, eave = client
    at = [round(v) for v in shot.pixel((20, 7, 0))]
    assert test_client.post(f"/photos/{photo_id}/wreaths", json={"point": at}).status_code == 409
    test_client.put(f"/photos/{photo_id}/scale", json={"measurement": {"region_id": str(store.items[0].id), "length_in": 480}})
    made = test_client.post(f"/photos/{photo_id}/wreaths", json={"point": at})
    assert made.status_code == 200, made.text
    wreath = next(r for r in made.json()["regions"] if r["kind"] == "wreath")
    assert wreath["source"] == "manual" and wreath["label"] == "Wreath, 36 in"
    small = store.masks[wreath["id"]].sum()
    resized = test_client.put(f"/photos/{photo_id}/scale/overrides/{wreath['id']}", json={"at": at, "size_in": 48})
    assert resized.status_code == 200
    assert store.masks[wreath["id"]].sum() == pytest.approx(small * (48 / 36) ** 2, rel=0.05)
    placed = test_client.get(f"/photos/{photo_id}/placement").json()["regions"][wreath["id"]]
    assert placed["count"] == 150 and placed["size_in"] == 48
    # A line's midpoint is refused on a wreath.
    assert test_client.put(f"/photos/{photo_id}/scale/overrides/{wreath['id']}", json={"mid": at}).status_code == 400
