"""M3 Day 2: scale and 3-D placement of roof lines (services/scale.py) on a synthetic house seen
by a known pinhole camera. A math check: every rule whose assumption holds must come out exact."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import camera as camera_module
from app.services import partition, scale
from app.services.scale import Line

W, H, F = 1600, 1200, 1300.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
FT = 12.0  # the house is drawn in feet; the solver works in inches


class Shot:
    """A camera looking at a house from the front-right, a little above the eaves."""

    def __init__(self, yaw_deg=-30.0, pitch_deg=6.0, centre=(55.0, 14.0, -60.0)):
        y, p = np.radians(yaw_deg), np.radians(pitch_deg)
        yaw = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
        tilt = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
        flip = np.diag([1.0, -1.0, 1.0])     # world y is up, image y is down
        self.R = tilt @ flip @ yaw
        self.C = np.array(centre)

    def cam(self, world):
        return self.R @ (np.asarray(world, float) - self.C)

    def pixel(self, world):
        p = K @ self.cam(world)
        return [float(p[0] / p[2]), float(p[1] / p[2])]

    def camera(self) -> camera_module.Camera:
        scale_, centre = max(W, H) / 2, np.array([W / 2, H / 2])

        def normalised(direction):
            p = K @ (self.R @ np.asarray(direction, float))
            return np.array([(p[0] - centre[0] * p[2]) / scale_, (p[1] - centre[1] * p[2]) / scale_, p[2]])

        scene = partition.Scene(segments=np.zeros((0, 4)), vertical=normalised([0, 1, 0]),
                                horizontal=[normalised([1, 0, 0]), normalised([0, 0, 1])],
                                centre=centre, scale=scale_)
        return camera_module.build(scene, (W, H), None)

    def line(self, name, a, b, edge, side):
        return Line(name, [self.pixel(a), self.pixel(b)], edge, side)


def true_length(a, b) -> float:
    return float(np.linalg.norm(np.subtract(b, a))) * FT


def facade_names(shot):
    cam = shot.camera()
    return cam, cam.facades[0], cam.facades[1]   # the x wall (front), the z wall (side)


def test_measured_wall_inherited_wall_gable_rakes_and_ridge_are_exact():
    shot = Shot()
    cam, front, side = facade_names(shot)
    # Side-gable house 40 x 30 ft, eaves at 10 ft, 6/12 roof: the right end is a gable.
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    rake_front = shot.line("rake1", (40, 10, 0), (40, 17.5, 15), "rake", None)
    rake_back = shot.line("rake2", (40, 17.5, 15), (40, 10, 30), "rake", None)
    ridge = shot.line("ridge", (0, 17.5, 15), (40, 17.5, 15), "ridge", front)
    lines = [eave, rake_front, rake_back, ridge]
    measurement = {"facade": front, "points": eave.points, "length_in": 40 * FT}
    solved = scale.solve(cam, lines, measurement)

    assert solved.walls[front]["source"] == "measured" and solved.walls[side]["source"] == "corner"
    got = solved.lines
    assert got["eave"].length_in == pytest.approx(40 * FT, rel=1e-6)
    for name, (a, b) in {"rake1": ((40, 10, 0), (40, 17.5, 15)), "rake2": ((40, 17.5, 15), (40, 10, 30))}.items():
        assert got[name].method == "gable" and got[name].plane == f"{side} gable wall"
        assert got[name].length_in == pytest.approx(true_length(a, b), rel=0.01)
    assert got["ridge"].method == "ridge" and got["ridge"].confidence == "exact"
    assert got["ridge"].length_in == pytest.approx(40 * FT, rel=0.01)
    # Positions, not just lengths: every placed point projects back onto its pixel.
    for line in lines:
        for pixel, point in zip(line.points, got[line.id].points3d):
            assert cam.project(point) == pytest.approx(pixel, abs=1.0)


def test_hip_roof_by_equal_pitch_and_its_ridge_are_exact_when_the_assumption_holds():
    shot = Shot()
    cam, front, side = facade_names(shot)
    # Hip roof on a 40 x 30 ft box, 6/12 on every face: hips rise over the corner bisectors.
    lines = [
        shot.line("front", (0, 10, 0), (40, 10, 0), "eave", front),
        shot.line("right", (40, 10, 0), (40, 10, 30), "eave", side),
        shot.line("hip", (40, 10, 0), (25, 17.5, 15), "hip", None),
        shot.line("ridge", (15, 17.5, 15), (25, 17.5, 15), "ridge", front),
    ]
    solved = scale.solve(cam, lines, {"facade": front, "points": lines[0].points, "length_in": 40 * FT})
    got = solved.lines
    assert got["right"].length_in == pytest.approx(30 * FT, rel=0.01)
    assert got["hip"].method == "hip" and got["hip"].confidence == "stated"
    assert "one pitch (implies 6.0/12)" in got["hip"].assumption
    assert got["hip"].length_in == pytest.approx(true_length((40, 10, 0), (25, 17.5, 15)), rel=0.01)
    assert got["ridge"].method == "ridge" and got["ridge"].length_in == pytest.approx(10 * FT, rel=0.01)


def test_a_ridge_with_no_anchor_takes_the_stated_pitch_from_the_nearest_eave():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    ridge = shot.line("ridge", (5, 17.5, 15), (35, 17.5, 15), "ridge", front)
    solved = scale.solve(cam, [eave, ridge], {"facade": front, "points": eave.points, "length_in": 40 * FT})
    got = solved.lines["ridge"]
    assert got.method == "pitch" and got.confidence == "assumed"
    assert got.assumption.startswith("depth from an assumed 6/12 pitch")
    assert got.length_in == pytest.approx(30 * FT, rel=0.01)
    # A photo whose planes estimated 8/12 says so, and a 6/12 roof then comes out wrong by that.
    other = scale.solve(cam, [eave, ridge], {"facade": front, "points": eave.points, "length_in": 40 * FT},
                        interpreted={"pitch": {"0": {"nearest_standard": {"rise": 8, "run": 12}}}})
    assert other.lines["ridge"].assumption.startswith("depth from an estimated 8/12 pitch")


def test_without_a_measurement_nothing_is_placed_and_the_reason_is_given():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    solved = scale.solve(cam, [shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)], None)
    assert solved.needs == "measurement" and solved.lines["eave"].needs == "measurement"


def test_override_fraction_is_even_at_the_middle_and_bunches_toward_the_far_end():
    assert scale.override_fraction(0.25, 0.5) == pytest.approx(0.25)
    # The real middle drawn at 0.6 of the line: the far end is compressed.
    assert scale.override_fraction(0.5, 0.6) == pytest.approx(0.6)
    fractions = [scale.override_fraction(t / 4, 0.6) for t in range(5)]
    assert fractions[0] == 0 and fractions[-1] == pytest.approx(1.0)
    steps = np.diff(fractions)
    assert all(a > b for a, b in zip(steps, steps[1:]))


def test_geometry_route_scales_from_a_saved_eave_and_overrides_survive_a_new_measurement(monkeypatch):
    import io
    import uuid
    from types import SimpleNamespace
    from unittest.mock import Mock

    import httpx
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from app.config import Settings
    from app.routers import photos
    from app.storage import get_store
    from test_roofline_v3 import MemoryStore

    monkeypatch.setattr(httpx, "post", Mock(side_effect=AssertionError("no network")))
    cfg = Settings(_env_file=None, segmentation_provider="mock", instruction_parser="rules")
    monkeypatch.setattr(photos, "get_settings", lambda: cfg)
    store = MemoryStore()
    api = FastAPI()
    api.include_router(photos.router)
    api.dependency_overrides[get_store] = lambda: store
    client = TestClient(api)
    buffer = io.BytesIO()
    Image.new("RGB", (W, H)).save(buffer, "PNG")
    photo_id = client.post("/photos", files={"file": ("h.png", buffer.getvalue(), "image/png")},
                           data={"prompt": ""}).json()["id"]

    shot = Shot()
    cam, front, side = facade_names(shot)
    store.row.interpreted = {"camera": cam.to_dict()}

    def region(line):
        return SimpleNamespace(id=uuid.uuid4(), kind="roof", geometry="line", edge=line.edge, side=line.side,
                               polyline=[[round(x), round(y)] for x, y in line.points], source="auto", role="target")

    eave = region(shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front))
    rake = region(shot.line("rake", (40, 10, 0), (40, 17.5, 15), "rake", None))
    odd = region(shot.line("odd", (10, 12, 0), (30, 12, 0), "unknown", None))
    store.items = [eave, rake, odd]

    before = client.get(f"/photos/{photo_id}/geometry").json()
    assert before["needs"] == "measurement"

    assert client.put(f"/photos/{photo_id}/scale/overrides/{odd.id}",
                      json={"mid": [700, 800], "length_in": 240}).status_code == 200
    saved = client.put(f"/photos/{photo_id}/scale", json={"measurement": {"region_id": str(eave.id), "length_in": 480}})
    assert saved.status_code == 200 and saved.json()["scale"]["overrides"][str(odd.id)]["length_in"] == 240

    solved = client.get(f"/photos/{photo_id}/geometry").json()
    assert solved["reference_edited"] is False
    lines = solved["lines"]
    assert solved["walls"][front]["source"] == "measured" and solved["walls"][side]["source"] == "corner"
    assert lines[str(eave.id)]["length_in"] == pytest.approx(480, rel=0.01)
    assert lines[str(rake.id)]["length_in"] == pytest.approx(true_length((40, 10, 0), (40, 17.5, 15)), rel=0.02)
    assert lines[str(odd.id)]["method"] == "override" and lines[str(odd.id)]["length_in"] == 240
    assert client.put(f"/photos/{photo_id}/scale/overrides/{odd.id}").status_code == 200
    assert client.get(f"/photos/{photo_id}/geometry").json()["lines"][str(odd.id)]["needs"] == "side"
    # Dragging the measured eave afterwards moves the scale with it, and the photo says so.
    eave.polyline = [[x + 6, y] for x, y in eave.polyline]
    assert client.get(f"/photos/{photo_id}/geometry").json()["reference_edited"] is True


def test_the_measured_line_reads_exactly_the_typed_length_on_the_wall_it_was_measured_against():
    """Regression (photo f943e6b3): a rake measured as 9 ft on the front wall read 7 ft 10 in,
    because the rake rule then put it on the other gable wall."""
    shot = Shot()
    cam, front, side = facade_names(shot)
    rake = shot.line("rake1", (40, 10, 0), (40, 17.5, 15), "rake", None)
    lines = [shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front), rake,
             shot.line("rake2", (40, 17.5, 15), (40, 10, 30), "rake", None)]
    solved = scale.solve(cam, lines, {"facade": front, "region_id": "rake1", "length_in": 108})
    got = solved.lines["rake1"]
    assert got.method == "measured" and got.wall == front
    assert got.length_in == pytest.approx(108, rel=1e-6)


def test_a_broken_off_piece_of_the_other_slope_does_not_decide_a_rakes_wall():
    shot = Shot()
    cam, front, side = facade_names(shot)
    rake = shot.line("rake", (40, 10, 0), (40, 17.5, 15), "rake", None)
    piece = shot.line("piece", (40, 17.5, 15), (40, 16.0, 18), "rake", None)   # a fifth of the other slope
    reach = scale.CORNER_REACH_FRACTION * max(W, H)
    alone = scale._gable_facade(cam, rake, [rake], reach, 4.0)
    assert scale._gable_facade(cam, rake, [rake, piece], reach, 4.0) == alone
    # Its whole other slope still pairs, and gives the gable wall.
    other = shot.line("other", (40, 17.5, 15), (40, 10, 30), "rake", None)
    assert scale._gable_facade(cam, rake, [rake, other], reach, 4.0) == side


def test_a_typed_length_far_from_what_the_photo_suggests_is_questioned():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    odd = shot.line("odd", (5, 12, 0), (35, 12, 0), "unknown", None)        # 30 ft along the front wall
    measurement = {"facade": front, "points": eave.points, "length_in": 480}
    slip = scale.solve(cam, [eave, odd], measurement, overrides={"odd": {"length_in": 24}}).lines["odd"]
    assert slip.length_in == 24 and slip.warning.startswith("2 ft seems short") and "30 ft" in slip.warning
    right = scale.solve(cam, [eave, odd], measurement, overrides={"odd": {"length_in": 31 * 12}}).lines["odd"]
    assert right.warning is None
