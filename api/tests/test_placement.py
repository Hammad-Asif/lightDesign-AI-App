"""M3 Day 3: bulbs along lit lines (services/placement.py), on the synthetic house and known camera
of test_scale.py. Positions are checked against the true 3-D points, projected."""
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.services import placement, scale
from test_scale import FT, Shot, facade_names

C9 = SimpleNamespace(id=uuid.uuid4(), category="c9", bulb_diameter_in=1.125, spacing_in=12.0, spacing_options_in=[12.0, 15.0, 18.0],
                     colours=[], twinkle=False, drops_in=[], diameter_in=None, bulb_count=None, strand_length_ft=None)
ICICLE = SimpleNamespace(id=uuid.uuid4(), category="icicle", bulb_diameter_in=0.2, spacing_in=6.0, spacing_options_in=[],
                         colours=["warm_white"], twinkle=False, drops_in=[12.0, 18.0, 24.0], diameter_in=None,
                         bulb_count=70, strand_length_ft=7.5)
CATALOGUE = {str(C9.id): C9, str(ICICLE.id): ICICLE}


def c9(spacing=12.0):
    return {"product_id": str(C9.id), "spacing_in": spacing,
            "pattern": {"steps": [{"colour": "red", "count": 1}, {"colour": "green", "count": 1}]}}


def region(shot, name, a, b, edge, side, hidden=None):
    line = shot.line(name, a, b, edge, side)
    return SimpleNamespace(id=name, kind="roof", geometry="line", edge=edge, side=side, hidden=hidden,
                           polyline=[list(p) for p in line.points]), line


def run(shot, cam, regions, settings, measurement, overrides=None):
    lines = [scale.Line(r.id, r.polyline, r.edge, r.side) for r in regions]
    solved = scale.solve(cam, lines, measurement, overrides=overrides)
    return placement.place(cam, regions, settings, CATALOGUE, solved, overrides)


def test_bulbs_sit_on_the_true_one_foot_marks_of_the_eave_and_shrink_with_depth():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave, line = region(shot, "eave", (0, 10, 0), (40, 10, 0), "eave", front)
    placed = run(shot, cam, [eave], {"eave": c9()}, {"facade": front, "points": line.points, "length_in": 40 * FT})
    bulbs = placed["eave"]["bulbs"]
    assert placed["eave"]["count"] == 41 and placed["eave"]["hidden"] == 0
    for k, bulb in enumerate(bulbs):
        assert bulb[:2] == pytest.approx(shot.pixel((k, 10, 0)), abs=1.0)
    # The eave's left end is farther from this camera than its right end.
    sizes = [bulb[2] for bulb in bulbs]
    assert sizes[0] < sizes[-1] and all(a <= b + 1e-9 for a, b in zip(sizes, sizes[1:]))
    assert bulbs[0][2] == pytest.approx(cam.focal * 1.125 / (shot.cam((0, 10, 0))[2] * FT), rel=0.01)
    assert [bulb[3] for bulb in bulbs[:2]] == ["#ff2a2a", "#1fd65f"]


def test_spacing_carries_round_a_same_level_corner_without_resetting():
    shot = Shot()
    cam, front, side = facade_names(shot)
    front_eave, line = region(shot, "front", (0, 10, 0), (40, 10, 0), "eave", front)
    side_eave, _ = region(shot, "side", (40, 10, 0), (40, 10, 30), "eave", side)
    placed = run(shot, cam, [front_eave, side_eave], {"front": c9(18), "side": c9(18)},
                 {"facade": front, "points": line.points, "length_in": 40 * FT})
    assert placed["front"]["path"] == placed["side"]["path"]
    assert placed["front"]["count"] + placed["side"]["count"] == int(70 * 12 // 18) + 1
    # 480 in is 26 bulbs of 18 in plus 12 in: the next bulb is 6 in round the corner, not at it.
    assert placed["side"]["bulbs"][0][:2] == pytest.approx(shot.pixel((40, 10, 0.5)), abs=1.0)


def test_an_eave_meeting_a_rake_ends_the_run():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    front_eave, line = region(shot, "front", (0, 10, 0), (40, 10, 0), "eave", front)
    rake, _ = region(shot, "rake", (40, 10, 0), (40, 17.5, 15), "rake", None)
    back, _ = region(shot, "back", (40, 17.5, 15), (40, 10, 30), "rake", None)
    placed = run(shot, cam, [front_eave, rake, back], {"front": c9(), "rake": c9(), "back": c9()},
                 {"facade": front, "points": line.points, "length_in": 40 * FT})
    assert placed["front"]["path"] != placed["rake"]["path"]
    # The rake starts its own spacing at its foot.
    assert placed["rake"]["bulbs"][0][:2] == pytest.approx(shot.pixel((40, 10, 0)), abs=1.0)


def test_a_roof_step_ends_the_run_even_where_the_ends_meet_in_the_photo():
    shot = Shot()
    cam = shot.camera()
    up = scale.world_up(cam)
    upper = placement.Member("upper", [[0.0, 0.0], [100.0, 0.0]], [np.zeros(3), np.array([100.0, 0, 0])], [])
    lower = placement.Member("lower", [[101.0, 0.0], [200.0, 0.0]],
                             [np.array([100.0, 0, 0]) - up * 24, np.array([200.0, 0, 0]) - up * 24], [])
    level = placement.Member("level", [[101.0, 0.0], [200.0, 0.0]], [np.array([100.0, 0, 0]), np.array([200.0, 0, 0])], [])
    edges = {"upper": "eave", "lower": "eave", "level": "eave"}
    assert len(placement._chain([upper, lower], edges, cam, up, 10.0)) == 2
    assert len(placement._chain([upper, level], edges, cam, up, 10.0)) == 1


def test_hidden_bulbs_are_counted_and_flagged_never_dropped():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    whole, line = region(shot, "eave", (0, 10, 0), (40, 10, 0), "eave", front)
    drawn = float(np.linalg.norm(np.subtract(line.points[1], line.points[0])))
    whole.hidden = [[0.4 * drawn, 0.6 * drawn]]
    placed = run(shot, cam, [whole], {"eave": c9()}, {"facade": front, "points": line.points, "length_in": 40 * FT})
    assert placed["eave"]["count"] == 41
    assert 0 < placed["eave"]["hidden"] < 41
    assert sum(1 for b in placed["eave"]["bulbs"] if b[4]) == placed["eave"]["hidden"]


def test_icicle_drops_every_6_in_cycling_12_18_24_and_hang_straight_down():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave, line = region(shot, "eave", (0, 10, 0), (10, 10, 0), "eave", front)
    placed = run(shot, cam, [eave], {"eave": {"product_id": str(ICICLE.id)}},
                 {"facade": front, "points": line.points, "length_in": 10 * FT})
    drops = placed["eave"]["drops"]
    assert len(drops) == 21
    for k, drop in enumerate(drops[:6]):
        length_ft = [12.0, 18.0, 24.0][k % 3] / FT
        assert drop[2:] == pytest.approx(shot.pixel((k * 0.5, 10 - length_ft, 0)), abs=1.0)
    per_drop = round(70 / (7.5 * 12 / 6))
    assert placed["eave"]["count"] == 21 * per_drop


def test_an_overridden_line_is_walked_by_the_midpoint_mapping():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave, line = region(shot, "eave", (0, 10, 0), (40, 10, 0), "eave", front)
    odd, odd_line = region(shot, "odd", (5, 12, 0), (35, 12, 0), "unknown", None)
    a, b = np.asarray(odd.polyline[0]), np.asarray(odd.polyline[1])
    overrides = {"odd": {"mid": list(a + 0.6 * (b - a)), "length_in": 30 * FT}}
    placed = run(shot, cam, [eave, odd], {"odd": c9(12)}, {"facade": front, "points": line.points, "length_in": 480},
                 overrides)
    bulbs = placed["odd"]["bulbs"]
    assert placed["odd"]["count"] == 31
    assert bulbs[0][:2] == pytest.approx(list(a), abs=0.5) and bulbs[-1][:2] == pytest.approx(list(b), abs=0.5)
    assert bulbs[15][:2] == pytest.approx(list(a + 0.6 * (b - a)), abs=0.5)


def test_a_line_that_cannot_be_placed_says_why():
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave, _ = region(shot, "eave", (0, 10, 0), (40, 10, 0), "eave", front)
    assert run(shot, cam, [eave], {"eave": c9()}, None)["eave"] == {"needs": "measurement"}


def test_garland_lights_every_4_in_alternating_across_a_10_in_band():
    garland = SimpleNamespace(id=uuid.uuid4(), category="garland", bulb_diameter_in=0.2, spacing_in=4.0, spacing_options_in=[],
                              colours=["warm_white"], twinkle=False, drops_in=[], diameter_in=10.0, bulb_count=50,
                              strand_length_ft=9.0)
    CATALOGUE[str(garland.id)] = garland
    shot = Shot()
    cam, front, _ = facade_names(shot)
    eave, line = region(shot, "eave", (0, 10, 0), (10, 10, 0), "eave", front)
    placed = run(shot, cam, [eave], {"eave": {"product_id": str(garland.id)}},
                 {"facade": front, "points": line.points, "length_in": 10 * FT})
    bulbs = placed["eave"]["bulbs"]
    assert placed["eave"]["count"] == 10 * 12 // 4 + 1
    # First bulb 3 in above the line (0.3 of the 10 in band), second 3 in below, 4 in along.
    assert bulbs[0][:2] == pytest.approx(shot.pixel((0, 10 + 3 / FT, 0)), abs=1.0)
    assert bulbs[1][:2] == pytest.approx(shot.pixel((4 / FT, 10 - 3 / FT, 0)), abs=1.0)
    band = placed["eave"]["band"][0]
    assert band[2] == pytest.approx(cam.focal * 10 / (shot.cam((0, 10, 0))[2] * FT), rel=0.01)
