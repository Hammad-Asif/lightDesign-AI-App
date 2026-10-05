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


@pytest.fixture
def flush_eaves(monkeypatch):
    """These houses are drawn with their eaves on the wall: no overhang for the solver to expect."""
    monkeypatch.setattr(scale, "EAVE_OVERHANG_IN", 0.0)


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

    # The gable end is a wall of its own ("right#1"): the far gable would be another, a house's width away.
    gable_wall = solved.lines["rake1"].wall
    assert solved.walls[front]["source"] == "measured" and solved.walls[gable_wall]["source"] == "corner"
    assert solved.walls[gable_wall]["facade"] == side
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
    solved = scale.solve(cam, lines, {"facade": front, "region_id": "rake1", "length_in": 108, "facade_given": True})
    got = solved.lines["rake1"]
    assert got.method == "measured"
    assert got.length_in == pytest.approx(108, rel=1e-6)
    # A rake lies in its gable wall whatever side was picked (photo 6c910b84: a gable-end rake
    # picked as the long side wall put the whole house on the wrong plane). Said, not hidden.
    assert solved.reference["facade"] == side and solved.reference["how"].startswith("gable (not the front side")


def test_a_broken_off_piece_of_the_other_slope_does_not_decide_a_rakes_wall():
    shot = Shot()
    cam, front, side = facade_names(shot)
    rake = shot.line("rake", (40, 10, 0), (40, 17.5, 15), "rake", None)
    piece = shot.line("piece", (40, 17.5, 15), (40, 16.0, 18), "rake", None)   # a fifth of the other slope
    reach = scale.CORNER_REACH_FRACTION * max(W, H)
    alone = scale._gable_facade(cam, rake, [rake], reach, 4.0, cam.facades, {})
    assert scale._gable_facade(cam, rake, [rake, piece], reach, 4.0, cam.facades, {}) == alone
    # Its whole other slope still pairs, and gives the gable wall.
    other = shot.line("other", (40, 17.5, 15), (40, 10, 30), "rake", None)
    assert scale._gable_facade(cam, rake, [rake, other], reach, 4.0, cam.facades, {}) == (side, "pair")


def test_a_typed_length_far_from_what_the_photo_suggests_is_questioned(flush_eaves):
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    odd = shot.line("odd", (5, 12, 0), (35, 12, 0), "unknown", None)        # 30 ft along the front wall
    measurement = {"facade": front, "points": eave.points, "length_in": 480}
    slip = scale.solve(cam, [eave, odd], measurement, overrides={"odd": {"length_in": 24}}).lines["odd"]
    assert slip.length_in == 24 and slip.warning.startswith("2 ft seems short") and "30 ft" in slip.warning
    right = scale.solve(cam, [eave, odd], measurement, overrides={"odd": {"length_in": 31 * 12}}).lines["odd"]
    assert right.warning is None


def _front_hip_roof(shot, front, *, corners_as="rake"):
    """A 40 x 30 ft hip roof seen nearly front-on: one front eave, both corners, the ridge."""
    return [shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front),
            shot.line("left", (0, 10, 0), (15, 17.5, 15), corners_as, None),
            shot.line("right", (40, 10, 0), (25, 17.5, 15), corners_as, None),
            shot.line("ridge", (15, 17.5, 15), (25, 17.5, 15), "ridge", front)]


@pytest.mark.parametrize("yaw, x", [(-8.0, 20.0), (-12.0, 30.0)])
def test_hip_corners_marked_as_rakes_are_placed_as_hips_and_their_ridge_is_not_exact(yaw, x, flush_eaves):
    """Regression (client test, hip roof): the corners came from detection as rakes. With no second
    rake at a peak they were put on a side wall (10 ft 5 in for a ~22 ft hip), and the ridge they
    met was called exact."""
    shot = Shot(yaw_deg=yaw, centre=(x, 14.0, -70.0))
    cam, front, _ = facade_names(shot)
    lines = _front_hip_roof(shot, front)
    reference = {"facade": front, "points": [shot.pixel((12, 4, 0)), shot.pixel((28, 4, 0))], "length_in": 16 * FT}
    got = scale.solve(cam, lines, reference).lines
    for name, (a, b) in {"left": ((0, 10, 0), (15, 17.5, 15)), "right": ((40, 10, 0), (25, 17.5, 15))}.items():
        assert got[name].method == "hip" and got[name].confidence == "stated"
        assert got[name].assumption.startswith("marked rake, but no second rake meets it at a peak")
        assert got[name].length_in == pytest.approx(true_length(a, b), rel=0.01)
    assert got["ridge"].plane == "between the hip tops" and got["ridge"].confidence == "stated"
    assert got["ridge"].length_in == pytest.approx(10 * FT, rel=0.01)


def test_a_small_move_of_a_ridge_end_does_not_swing_its_length():
    shot = Shot(yaw_deg=-8.0, centre=(20.0, 14.0, -70.0))
    cam, front, _ = facade_names(shot)
    lines = _front_hip_roof(shot, front)
    reference = {"facade": front, "points": [shot.pixel((12, 4, 0)), shot.pixel((28, 4, 0))], "length_in": 16 * FT}
    before = scale.solve(cam, lines, reference).lines["ridge"]
    reach = scale.CORNER_REACH_FRACTION * max(W, H)
    ridge = lines[3]
    span = ridge.points[1][0] - ridge.points[0][0]

    def moved(dx):
        line = Line("ridge", [[ridge.points[0][0] + dx, ridge.points[0][1]], ridge.points[1]], "ridge", front)
        return scale.solve(cam, lines[:3] + [line], reference).lines["ridge"]

    # Within reach of the hip top the end still hangs from it: nothing changes.
    for dx in (-0.5 * reach, 0.5 * reach):
        after = moved(dx)
        assert after.confidence == before.confidence == "stated"
        assert after.length_in == pytest.approx(before.length_in, rel=1e-6)
    # Past it, the length follows the move, by as much as the move is, and no more.
    after = moved(1.5 * reach)
    assert after.confidence == "stated"
    assert after.length_in == pytest.approx(before.length_in * (span - 1.5 * reach) / span, rel=0.05)


def test_a_gable_peak_still_makes_its_ridge_exact_and_a_lone_rake_top_does_not():
    shot = Shot()
    cam, front, side = facade_names(shot)
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    rake1 = shot.line("rake1", (40, 10, 0), (40, 17.5, 15), "rake", None)
    rake2 = shot.line("rake2", (40, 17.5, 15), (40, 10, 30), "rake", None)
    ridge = shot.line("ridge", (0, 17.5, 15), (40, 17.5, 15), "ridge", front)
    measurement = {"facade": front, "points": eave.points, "length_in": 40 * FT}
    peak = scale.solve(cam, [eave, rake1, rake2, ridge], measurement).lines
    assert peak["rake1"].method == "gable" and peak["ridge"].confidence == "exact"
    # The far rake hidden: no peak in view, so the ridge is not exact any more. The near rake runs
    # straight back along the side wall, which as a hip would imply an impossible pitch: it stays
    # a rake on that wall, and still measures right.
    lone = scale.solve(cam, [eave, rake1, ridge], measurement).lines
    assert lone["rake1"].method == "gable" and lone["rake1"].wall.split("#")[0] == side
    assert lone["rake1"].length_in == pytest.approx(true_length((40, 10, 0), (40, 17.5, 15)), rel=0.01)
    assert lone["ridge"].confidence != "exact"


def test_a_wall_whose_normal_faces_away_still_puts_the_house_in_front_of_the_camera():
    """Regression (photos 14851699, 38623ee6): with the measured wall's normal pointing away from
    the camera, every point came out behind it. Lengths survived the mirror; which end of a line
    is its top did not, so ridges hung from rake feet."""
    shot = Shot()
    cam, front, _ = facade_names(shot)
    flipped = camera_module.Camera.from_dict(cam.to_dict())
    flipped.horizontal = [-np.asarray(v, float) if i == 0 else v for i, v in enumerate(flipped.horizontal)]
    assert float(flipped.wall_normal(front) @ cam.wall_normal(front)) < 0
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    for camera in (cam, flipped):
        got = scale.solve(camera, [eave], {"facade": front, "points": eave.points, "length_in": 40 * FT})
        points = got.lines["eave"].points3d
        assert all(p[2] > 0 for p in points)
        assert got.lines["eave"].length_in == pytest.approx(40 * FT, rel=1e-6)


def test_a_wall_is_not_inherited_from_corners_that_put_it_through_the_camera():
    shot = Shot()
    cam, front, side = facade_names(shot)
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    # Two lines claimed for the side wall, meeting the front eave at both of its ends: only one of
    # those corners can be on the side wall, and the two readings straddle the camera.
    a = Line("a", [eave.points[0], [eave.points[0][0] - 40, eave.points[0][1] - 60]], "junction", side)
    b = Line("b", [eave.points[1], [eave.points[1][0] + 40, eave.points[1][1] - 60]], "junction", side)
    solved = scale.solve(cam, [eave, a, b], {"facade": front, "points": eave.points, "length_in": 40 * FT})
    wall = solved.walls.get(side)
    assert wall is None or abs(wall["distance_in"]) > 1.0


def test_a_reference_that_makes_the_house_implausibly_wide_is_flagged(flush_eaves):
    """Client test: a window about 6 ft wide typed as 16 ft made the one-storey front 112 ft."""
    shot = Shot(yaw_deg=-8.0, centre=(20.0, 14.0, -70.0))
    cam, front, _ = facade_names(shot)
    lines = _front_hip_roof(shot, front, corners_as="hip")
    window = [shot.pixel((17, 6, 0)), shot.pixel((23, 6, 0))]       # 6 ft wide
    right = scale.solve(cam, lines, {"facade": front, "points": window, "length_in": 6 * FT})
    assert right.warning is None and right.to_dict()["warning"] is None
    wrong = scale.solve(cam, lines, {"facade": front, "points": window, "length_in": 16 * FT})
    assert wrong.warning.startswith("This measurement makes the front of the house 107 ft wide")
    # Too small is not flagged: the eaves in view may be one short piece of the house.
    assert scale.solve(cam, lines, {"facade": front, "points": window, "length_in": 12}).warning is None


OH = 1.5   # ft: the house below's eaves overhang their walls by scale.EAVE_OVERHANG_IN


def _overhung_gable(shot, front):
    """40 ft front eave hanging 18 in in front of the front wall (z = 0), its gable end at x = 40."""
    return [shot.line("eave", (0, 10, -OH), (40, 10, -OH), "eave", front),
            shot.line("rake1", (40, 10, -OH), (40, 17.5, 15), "rake", None),
            shot.line("rake2", (40, 17.5, 15), (40, 10, 31.5), "rake", None)]


def test_an_eave_measured_against_a_wall_feature_allows_for_its_overhang(monkeypatch):
    shot = Shot()
    cam, front, _ = facade_names(shot)
    lines = _overhung_gable(shot, front)
    garage_door = {"facade": front, "points": [shot.pixel((4, 2, 0)), shot.pixel((20, 2, 0))], "length_in": 16 * FT}
    got = scale.solve(cam, lines, garage_door).lines
    assert got["eave"].length_in == pytest.approx(40 * FT, rel=1e-3)
    assert "eave overhang assumed to be 18 in" in got["eave"].assumption
    for name, (a, b) in {"rake1": ((40, 10, -OH), (40, 17.5, 15)), "rake2": ((40, 17.5, 15), (40, 10, 31.5))}.items():
        assert got[name].length_in == pytest.approx(true_length(a, b), rel=0.01)
    # Without the overhang the eave is put on the wall, behind where it hangs, and reads long.
    monkeypatch.setattr(scale, "EAVE_OVERHANG_IN", 0.0)
    assert scale.solve(cam, lines, garage_door).lines["eave"].length_in > 40 * FT * 1.02


@pytest.mark.parametrize("by", ["points", "line"])
def test_a_measurement_along_the_eave_puts_the_wall_the_overhang_behind_it(by):
    shot = Shot()
    cam, front, _ = facade_names(shot)
    lines = _overhung_gable(shot, front)
    # Where a porch roof meets the front wall: on the wall itself, 30 ft long.
    junction = shot.line("junction", (5, 8, 0), (35, 8, 0), "junction", front)
    measurement = ({"facade": front, "points": lines[0].points, "length_in": 40 * FT} if by == "points"
                   else {"facade": front, "region_id": "eave", "length_in": 40 * FT})
    got = scale.solve(cam, lines + [junction], measurement).lines
    assert got["eave"].length_in == pytest.approx(40 * FT, rel=1e-6) and got["eave"].assumption is None
    assert got["junction"].length_in == pytest.approx(30 * FT, rel=1e-3)
    assert "eave overhang assumed" in got["junction"].assumption


def test_an_unanchored_ridge_takes_the_pitch_measured_on_the_gable_rakes():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    # Side-gable 8/12 roof: the rakes rise 10 ft over 15 ft. The ridge piece in view reaches
    # neither peak, so rule (c) places it by the pitch.
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    rakes = [shot.line("rake1", (40, 10, 0), (40, 20, 15), "rake", None),
             shot.line("rake2", (40, 20, 15), (40, 10, 30), "rake", None)]
    ridge = shot.line("ridge", (5, 20, 15), (35, 20, 15), "ridge", front)
    measurement = {"facade": front, "points": eave.points, "length_in": 40 * FT}
    # The photo's planes estimated 6/12: the rakes' measurement wins.
    solved = scale.solve(cam, [eave, *rakes, ridge], measurement,
                         interpreted={"pitch": {"0": {"nearest_standard": {"rise": 6, "run": 12}}}})
    assert solved.pitch == {"rise": 8.0, "source": "measured"}
    got = solved.lines["ridge"]
    assert got.method == "pitch" and got.confidence == "stated"
    assert got.assumption.startswith("depth from the 8.0/12 pitch measured on the gable rakes")
    assert got.length_in == pytest.approx(30 * FT, rel=0.01)
    # With no rakes there is nothing to measure: the estimate stands, and is only assumed.
    alone = scale.solve(cam, [eave, ridge], measurement,
                        interpreted={"pitch": {"0": {"nearest_standard": {"rise": 6, "run": 12}}}})
    assert alone.pitch == {"rise": 6.0, "source": "estimated"} and alone.lines["ridge"].confidence == "assumed"


# -- a corner photo the camera read as one facade (photo 824fd558) ------------------------------------

F_CORNER = 2000.0      # a 45 mm lens on this 1600 px frame, far from the 26 mm the camera assumes
K_CORNER = np.array([[F_CORNER, 0, W / 2], [0, F_CORNER, H / 2], [0, 0, 1]])


class CornerShot(Shot):
    """From the front-left, the camera nearly level, through a longer lens than a phone's main one."""

    def __init__(self):
        super().__init__(yaw_deg=35.0, pitch_deg=1.0, centre=(-40.0, 6.0, -70.0))

    def pixel(self, world):
        p = K_CORNER @ self.cam(world)
        return [float(p[0] / p[2]), float(p[1] / p[2])]

    def one_facade_camera(self) -> camera_module.Camera:
        """What detection gave on the photo: the front's vanishing point and the vertical only."""
        scale_, centre = max(W, H) / 2, np.array([W / 2, H / 2])

        def normalised(direction):
            p = K_CORNER @ (self.R @ np.asarray(direction, float))
            return np.array([(p[0] - centre[0] * p[2]) / scale_, (p[1] - centre[1] * p[2]) / scale_, p[2]])

        scene = partition.Scene(segments=np.zeros((0, 4)), vertical=normalised([0, 1, 0]),
                                horizontal=[normalised([1, 0, 0])], centre=centre, scale=scale_)
        return camera_module.build(scene, (W, H), None)


def _corner_house(shot, front):
    """A 40 x 30 ft house with a side gable on its left wall (x = 0), eaves at 10 ft."""
    return [shot.line("front", (0, 10, 0), (40, 10, 0), "eave", front),
            shot.line("left_eave", (0, 10, 0), (0, 10, 30), "eave", None),
            shot.line("left_rake1", (0, 10, 0), (0, 17.5, 15), "rake", None),
            shot.line("left_rake2", (0, 17.5, 15), (0, 10, 30), "rake", None),
            shot.line("porch", (0, 4, 8), (0, 4, 26), "eave", None)]


def test_a_corner_photo_read_as_one_facade_gets_its_side_wall_and_lens_from_the_roof_lines(flush_eaves):
    """Regression (photo 824fd558): one facade found, the lens assumed at 26 mm, and the side
    gable's eave and rakes left with no wall. The side wall's own eaves give its vanishing point,
    and with it the focal length."""
    shot = CornerShot()
    cam = shot.one_facade_camera()
    assert cam.focal_source == "assumed" and len(cam.facades) == 1
    front = cam.facades[0]
    lines = _corner_house(shot, front)
    solved = scale.solve(cam, lines, {"facade": front, "region_id": "front", "length_in": 40 * FT, "facade_given": True})
    assert solved.camera_check["used"].startswith("second facade from the roof lines")
    used = solved.camera
    assert used.focal_source == "vp"
    assert used.focal == pytest.approx(F_CORNER, rel=0.03)
    got = solved.lines
    assert got["left_eave"].length_in == pytest.approx(30 * FT, rel=0.02)
    for name, (a, b) in {"left_rake1": ((0, 10, 0), (0, 17.5, 15)), "left_rake2": ((0, 17.5, 15), (0, 10, 30))}.items():
        assert got[name].method == "gable"
        assert got[name].length_in == pytest.approx(true_length(a, b), rel=0.03)


def test_one_side_eave_alone_needs_a_second_reading_before_it_moves_the_camera():
    shot = CornerShot()
    cam = shot.one_facade_camera()
    front = cam.facades[0]
    lone = [line for line in _corner_house(shot, front) if line.id != "porch"]
    # As on the photo, the camera set the vertical pair aside, and it read the true lens.
    assert cam.focal_checks["vp_rejected"][0]["focal_35mm"] == pytest.approx(
        camera_module._to_35mm(F_CORNER, max(W, H)), rel=0.03)
    # One eave, nothing backing it: no camera is built from it.
    cam.focal_checks.pop("vp_rejected")
    assert scale._focal_from_lines(cam, lone, 3.0) is None
    # Backed by a vanishing-point pair the conditioning test set aside, within 10%: taken.
    true_35 = camera_module._to_35mm(F_CORNER, max(W, H))
    cam.focal_checks["vp_rejected"] = [{"focal_35mm": round(true_35 * 1.05, 1), "spread": 0.26, "with": "vertical"}]
    focal_35mm, evidence = scale._focal_from_lines(cam, lone, 3.0)
    assert focal_35mm == pytest.approx(true_35, rel=0.03) and evidence["lines"] == ["left_eave"]
    # Backed by one 30% off: not taken.
    cam.focal_checks["vp_rejected"] = [{"focal_35mm": round(true_35 * 1.3, 1), "spread": 0.26, "with": "vertical"}]
    assert scale._focal_from_lines(cam, lone, 3.0) is None


def test_two_gables_meeting_at_an_inside_corner_keep_their_rakes_as_rakes():
    """Regression (photo 824fd558, 7e60f49b): where a side gable meets a front gable, both rakes'
    feet sit where two walls' eaves meet. Each comes down from its own gable peak, so neither is a
    hip."""
    rakes = [Line("side_rake", [[100, 200], [200, 100]], "rake", None),
             Line("side_peak", [[200, 100], [300, 200]], "rake", None),
             Line("front_rake", [[300, 200], [400, 90]], "rake", None),
             Line("front_peak", [[400, 90], [500, 200]], "rake", None)]
    eaves = [Line("side_eave", [[100, 200], [300, 200]], "eave", "left"),
             Line("front_eave", [[300, 200], [500, 205]], "eave", "front")]
    facade_of = {"side_eave": "left", "front_eave": "front"}
    assert scale._hip_readings(rakes + eaves, facade_of, 10.0, None) == {}
    # A lone rake at the same corner, no peak at its top: still the corner of a hip roof.
    hip = Line("hip", [[300, 200], [380, 120]], "rake", None)
    assert scale._hip_readings([hip] + eaves, facade_of, 10.0, None) == {"hip": "corner"}


def test_a_reference_that_shrinks_the_whole_house_is_flagged(flush_eaves):
    """Regression (photo 824fd558): a 12 in reference made a two-storey front 19 ft and every rake
    5 ft 6 in or under, and nothing said so."""
    shot = Shot()
    cam, front, side = facade_names(shot)
    lines = [shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front),
             shot.line("rake1", (40, 10, 0), (40, 17.5, 15), "rake", None),
             shot.line("rake2", (40, 17.5, 15), (40, 10, 30), "rake", None)]
    right = scale.solve(cam, lines, {"facade": front, "region_id": "eave", "length_in": 40 * FT, "facade_given": True})
    assert right.warning is None
    shrunk = scale.solve(cam, lines, {"facade": front, "region_id": "eave", "length_in": 10 * FT, "facade_given": True})
    assert shrunk.warning.startswith("This measurement makes the house at most 10 ft wide and its longest gable rake")
    assert "smaller than a house usually is" in shrunk.warning


def _dormer_house(shot, front, side):
    """A 40 ft front eave at 10 ft, the roof rising 6/12 behind it, and a gabled dormer standing on
    that roof: its front wall 4 ft back (meeting the roof at 12 ft), 16 ft wide, eaves at 16 ft and
    a 6/12 gable; its side walls run back 8 ft to where the roof reaches 16 ft. (Photo 6c910b84.)"""
    return [shot.line("front", (0, 10, 0), (40, 10, 0), "eave", front),
            shot.line("base", (10, 12, 4), (26, 12, 4), "junction", front),
            shot.line("d_rake1", (10, 16, 4), (18, 20, 4), "rake", None),
            shot.line("d_rake2", (18, 20, 4), (26, 16, 4), "rake", None),
            shot.line("cheek_left", (10, 16, 4), (10, 16, 12), "eave", side),
            shot.line("cheek_right", (26, 16, 4), (26, 16, 12), "eave", side),
            shot.line("valley", (26, 16, 12), (26, 12, 4), "valley", None)]


def test_a_dormer_standing_on_the_roof_is_measured_on_its_own_walls(flush_eaves):
    """Regression (photo 6c910b84): the dormers' rakes had no wall, their side eaves all shared
    one plane (ending 29 to 32 ft off the dormers), and a valley read twice its length off the
    nearest eave's face."""
    shot = Shot(yaw_deg=-30.0, pitch_deg=-8.0, centre=(55.0, 5.0, -55.0))   # standing, looking up at it
    cam, front, side = facade_names(shot)
    lines = _dormer_house(shot, front, side)
    solved = scale.solve(cam, lines, {"facade": front, "region_id": "front", "length_in": 40 * FT, "facade_given": True})
    got = solved.lines
    truth = {"d_rake1": ((10, 16, 4), (18, 20, 4)), "d_rake2": ((18, 20, 4), (26, 16, 4)),
             "cheek_left": ((10, 16, 4), (10, 16, 12)), "cheek_right": ((26, 16, 4), (26, 16, 12)),
             "valley": ((26, 16, 12), (26, 12, 4))}
    for name, (a, b) in truth.items():
        assert got[name].length_in == pytest.approx(true_length(a, b), rel=0.03), name
    assert got["d_rake1"].method == "gable" and got["valley"].method == "valley"
    # The two side walls face one way a dormer's width apart: two planes, not one.
    assert solved.walls and len({k for k in solved.walls if k.startswith(f"{side}@")}) == 2
    assert solved.checks == []


def test_a_short_rake_of_an_asymmetric_gable_takes_the_gable_wall_of_its_peak():
    """Photo 6c910b84: the gable's left rake is 477 px, its right 1233 px (it runs on down over
    the porch), too unequal to pair; the short one still stands in the same gable wall."""
    shot = Shot()
    cam, front, side = facade_names(shot)
    porch = shot.line("porch", (40, 6, 38), (60, 6, 38), "eave", front)   # the porch eave at the long rake's foot
    long = shot.line("long", (40, 17.5, 15), (40, 6, 38), "rake", None)    # down over the porch
    short = shot.line("short", (40, 17.5, 15), (40, 14, 8), "rake", None)  # cut short by a tree
    lines = [porch, short, long]
    reach = scale.CORNER_REACH_FRACTION * max(W, H)
    assert scale._rake_pairs(short, lines, reach) == []
    # Alone, the short rake has no gable wall; through the peak it shares, the long one's.
    assert scale._gable_facade(cam, short, lines, reach, 3.0, [front, side], {"porch": front}, follow_peak=False) == (None, None)
    assert scale._gable_facade(cam, long, lines, reach, 3.0, [front, side], {"porch": front}) == (side, "elimination")
    assert scale._gable_facade(cam, short, lines, reach, 3.0, [front, side], {"porch": front}) == (side, "peak")


def test_a_side_wall_reached_only_through_a_front_gable_on_the_front_wall_is_measured(flush_eaves):
    """Regression (photo 7d231dc3): the front gables stand on the front wall (their rakes' feet on
    the front eave), and the side gable's eave meets only a front gable's foot. Parallel walls share
    no corner, so the front gables' wall had no distance, and the side gable none either."""
    shot = CornerShot()
    cam = shot.one_facade_camera()
    front = cam.facades[0]
    lines = [shot.line("front", (10, 10, 0), (40, 10, 0), "eave", front),
             shot.line("cross1", (0, 10, 0), (5, 13, 0), "rake", None),       # a cross gable on the front wall
             shot.line("cross2", (5, 13, 0), (10, 10, 0), "rake", None),
             shot.line("left_eave", (0, 10, 0), (0, 10, 30), "eave", None),
             shot.line("left_rake1", (0, 10, 0), (0, 17.5, 15), "rake", None),
             shot.line("left_rake2", (0, 17.5, 15), (0, 10, 30), "rake", None),
             shot.line("porch", (0, 4, 8), (0, 4, 26), "eave", None)]
    solved = scale.solve(cam, lines, {"facade": front, "region_id": "front", "length_in": 30 * FT, "facade_given": True})
    got = solved.lines
    assert got["cross1"].length_in == pytest.approx(true_length((0, 10, 0), (5, 13, 0)), rel=0.03)
    assert got["left_eave"].length_in == pytest.approx(30 * FT, rel=0.03)
    assert got["left_rake1"].length_in == pytest.approx(true_length((0, 10, 0), (0, 17.5, 15)), rel=0.03)


def test_a_line_no_rule_places_gets_an_approximate_length_from_the_house_around_it(flush_eaves):
    """Where the geometry cannot place a line (an unknown edge, a scrap no corner joins, a length too
    unsure to show), its length is read at the scale of the placed lines beside it: approximate,
    marked assumed, and saying where it came from. Lines the geometry placed do not move."""
    shot = Shot()
    cam, front, side = facade_names(shot)
    eave = shot.line("eave", (0, 10, 0), (40, 10, 0), "eave", front)
    trim = shot.line("trim", (10, 12, 0), (30, 12, 0), "unknown", None)     # 20 ft along the front wall
    measurement = {"facade": front, "region_id": "eave", "length_in": 40 * FT, "facade_given": True}
    alone = scale.solve(cam, [eave], measurement).lines["eave"]
    solved = scale.solve(cam, [eave, trim], measurement)
    got = solved.lines["trim"]
    assert got.method == "estimate" and got.confidence == "assumed" and got.needs is None
    assert got.length_in == pytest.approx(20 * FT, rel=0.10)
    assert "read at the scale of the eave" in got.assumption
    assert solved.lines["eave"].length_in == pytest.approx(alone.length_in, rel=1e-9)
    # Nothing measured yet: nothing to read a scale from, so no estimate either.
    assert scale.solve(cam, [eave, trim], None).lines["trim"].length_in is None
