"""M3 Day 1: the pinhole camera (services/camera.py) and the scale measurement route. Offline."""
import io
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import Settings
from app.routers import photos
from app.services import camera, partition
from app.storage import exif_focal_35mm, get_store
from test_roofline_v3 import MemoryStore

W, H, F = 1600, 1200, 1400.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    yaw = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    pitch = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return pitch @ yaw


def scene_for(directions: list, vertical=None) -> partition.Scene:
    """A Scene whose vanishing points are the images of these 3-D directions under K."""
    scale, centre = max(W, H) / 2, np.array([W / 2, H / 2])

    def normalised(d):
        p = K @ np.asarray(d, float)
        return np.array([(p[0] - centre[0] * p[2]) / scale, (p[1] - centre[1] * p[2]) / scale, p[2]])

    return partition.Scene(segments=np.zeros((0, 4)), vertical=normalised(vertical) if vertical is not None else None,
                           horizontal=[normalised(d) for d in directions], centre=centre, scale=scale)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("no network in these tests")))


def test_focal_length_comes_from_two_perpendicular_vanishing_points():
    R = rotation(35, 8)
    cam = camera.build(scene_for([R @ [1, 0, 0], R @ [0, 0, 1]], R @ [0, 1, 0]), (W, H), None)
    assert cam.focal_source == "vp"
    assert cam.focal == pytest.approx(F, rel=1e-6)


def test_exif_is_used_only_without_a_vanishing_point_estimate_and_else_a_labelled_assumption():
    front_on = partition.Scene(segments=np.zeros((0, 4)), vertical=None, horizontal=[np.array([1.0, 0.0, 0.0])],
                               centre=np.array([W / 2, H / 2]), scale=max(W, H) / 2)
    assert camera.build(front_on, (W, H), 28.0).focal_source == "exif"
    assumed = camera.build(front_on, (W, H), None)
    assert assumed.focal_source == "assumed" and assumed.focal_35mm == pytest.approx(camera.ASSUMED_FOCAL_35MM)
    R = rotation(35, 8)
    both = camera.build(scene_for([R @ [1, 0, 0], R @ [0, 0, 1]], R @ [0, 1, 0]), (W, H), F / W * 36)
    assert both.focal_source == "vp" and both.focal_checks["agree"] is True


def test_wall_planes_and_pixel_round_trip_are_exact():
    R = rotation(-28, 5)
    level, other, up = R @ [1, 0, 0], R @ [0, 0, 1], R @ [0, 1, 0]
    cam = camera.build(scene_for([level, other], up), (W, H), None)
    for facade, direction in zip(cam.facades, (level, other)):
        truth = np.cross(direction, up)
        normal = cam.wall_normal(facade)
        assert abs(abs(float(normal @ truth)) / np.linalg.norm(truth) - 1) < 1e-9
    normal = cam.wall_normal(cam.facades[0])
    point = cam.on_wall([640, 410], normal, 9.5)
    assert float(normal @ point) == pytest.approx(9.5)
    assert cam.project(point) == pytest.approx([640, 410])


def test_facade_of_a_segment_follows_its_vanishing_point_not_its_slope():
    R = rotation(35, 0)
    cam = camera.build(scene_for([R @ [1, 0, 0], R @ [0, 0, 1]], R @ [0, 1, 0]), (W, H), None)
    toward = np.array(cam.horizontal[0][:2])
    a = np.array([800.0, 700.0])
    b = a + 200 * (toward - a) / np.linalg.norm(toward - a)
    assert cam.facade_of(a, b, 2.0) == cam.facades[0]
    # The same slope, elsewhere in the photo, does not run to that vanishing point.
    assert cam.facade_of(a + [0, -500], b + [0, -500], 2.0) is None


def test_exif_focal_is_read_from_the_upload_bytes():
    ex = Image.Exif()
    ex.get_ifd(0x8769)[41989] = 26
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, "JPEG", exif=ex)
    assert exif_focal_35mm(Image.open(io.BytesIO(buffer.getvalue()))) == 26.0
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, "PNG")
    assert exif_focal_35mm(Image.open(io.BytesIO(buffer.getvalue()))) is None


class Store(MemoryStore):
    def photo_dir(self, photo_id):
        return Path("/nonexistent")


@pytest.fixture
def scale_client(monkeypatch):
    cfg = Settings(_env_file=None, segmentation_provider="mock", instruction_parser="rules")
    monkeypatch.setattr(photos, "get_settings", lambda: cfg)
    store = Store()
    api = FastAPI()
    api.include_router(photos.router)
    api.dependency_overrides[get_store] = lambda: store
    client = TestClient(api)
    buffer = io.BytesIO()
    Image.new("RGB", (W, H)).save(buffer, "PNG")
    photo_id = client.post("/photos", files={"file": ("h.png", buffer.getvalue(), "image/png")},
                           data={"prompt": ""}).json()["id"]
    R = rotation(35, 0)
    cam = camera.build(scene_for([R @ [1, 0, 0], R @ [0, 0, 1]], R @ [0, 1, 0]), (W, H), None)
    store.row.interpreted = {"camera": cam.to_dict()}
    return client, store, photo_id, cam


def test_scale_measurement_reads_its_facade_from_the_vanishing_points(scale_client):
    client, store, photo_id, cam = scale_client
    toward = np.array(cam.horizontal[0][:2])
    a = np.array([800.0, 700.0])
    b = a + 300 * (toward - a) / np.linalg.norm(toward - a)
    result = client.put(f"/photos/{photo_id}/scale", json={"measurement": {
        "points": [[int(a[0]), int(a[1])], [int(round(b[0])), int(round(b[1]))]], "length_in": 192}})
    assert result.status_code == 200, result.text
    assert result.json()["scale"]["measurement"]["facade"] == cam.facades[0]


def test_scale_measurement_asks_for_a_side_it_cannot_read_and_refuses_a_missing_wall(scale_client):
    client, _, photo_id, cam = scale_client
    vertical = {"points": [[800, 500], [800, 700]], "length_in": 80}
    assert client.put(f"/photos/{photo_id}/scale", json={"measurement": vertical}).status_code == 400
    missing = ({"front", "left", "right"} - set(cam.facades)).pop()
    assert client.put(f"/photos/{photo_id}/scale",
                      json={"measurement": {**vertical, "facade": missing}}).status_code == 400
    saved = client.put(f"/photos/{photo_id}/scale", json={"measurement": {**vertical, "facade": cam.facades[1]}})
    assert saved.status_code == 200 and saved.json()["scale"]["measurement"]["length_in"] == 80
    cleared = client.put(f"/photos/{photo_id}/scale", json={"measurement": None})
    assert cleared.status_code == 200 and cleared.json()["scale"] is None
