"""Hand-drawn roof lines (services/drawn.py): ends join the lines already there, and the type and
side suggested for the picker, on the synthetic hip-roof house of test_scale."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.services import drawn, scale
from app.services.scale import Line
from test_scale import FT, H, W, Shot, facade_names, true_length

SIZE = (W, H)


def _house():
    shot = Shot(yaw_deg=-8.0, centre=(20.0, 14.0, -70.0))
    cam, front, _ = facade_names(shot)
    px = shot.pixel
    lines = {
        "eave_left": ([px((0, 10, 0)), px((14, 10, 0))], "eave", front),       # detection stopped at 14 ft...
        "eave_right": ([px((26, 10, 0)), px((40, 10, 0))], "eave", front),     # ...and started again at 26
        "hip_left": ([px((0, 10, 0)), px((15, 17.5, 15))], "hip", None),
        "hip_right": ([px((40, 10, 0)), px((25, 17.5, 15))], "hip", None),
    }
    return shot, cam, front, lines


def _nudge(pixel, dx=6.0, dy=-4.0):
    return [pixel[0] + dx, pixel[1] + dy]       # a hand never lands on the pixel


def test_a_filled_eave_gap_joins_both_sides_and_is_an_eave_of_their_side():
    shot, cam, front, lines = _house()
    drawn_line = [_nudge(shot.pixel((14, 10, 0))), _nudge(shot.pixel((26, 10, 0)), -5, 3)]
    snapped = drawn.snap(drawn_line, [points for points, _, _ in lines.values()], SIZE)
    assert snapped == [lines["eave_left"][0][1], lines["eave_right"][0][0]]
    assert drawn.suggest(snapped, list(lines.values()), cam, SIZE) == ("eave", front)


def test_a_ridge_drawn_between_the_hip_tops_is_a_ridge_and_measures_right():
    shot, cam, front, lines = _house()
    drawn_line = drawn.snap([_nudge(shot.pixel((15, 17.5, 15))), _nudge(shot.pixel((25, 17.5, 15)))],
                            [points for points, _, _ in lines.values()], SIZE)
    edge, side = drawn.suggest(drawn_line, list(lines.values()), cam, SIZE)
    assert (edge, side) == ("ridge", front)
    # With that type the solver hangs it from both hip tops: the real 10 ft.
    solve_lines = [Line(k, p, e, s) for k, (p, e, s) in lines.items()] + [Line("drawn", drawn_line, edge, side)]
    eave = lines["eave_left"][0]
    got = scale.solve(cam, solve_lines, {"facade": front, "points": eave, "length_in": 14 * FT}).lines["drawn"]
    assert got.plane == "between the hip tops"
    assert got.length_in == pytest.approx(10 * FT, rel=0.02)


def test_a_hip_drawn_from_an_eave_corner_to_a_ridge_end_is_a_hip():
    shot, cam, front, lines = _house()
    lines["ridge"] = ([shot.pixel((15, 17.5, 15)), shot.pixel((25, 17.5, 15))], "ridge", front)
    del lines["hip_right"]
    drawn_line = drawn.snap([_nudge(shot.pixel((40, 10, 0))), _nudge(shot.pixel((25, 17.5, 15)))],
                            [points for points, _, _ in lines.values()], SIZE)
    assert drawn.suggest(drawn_line, list(lines.values()), cam, SIZE) == ("hip", None)
    solve_lines = [Line(k, p, e, s) for k, (p, e, s) in lines.items()] + [Line("drawn", drawn_line, "hip", None)]
    got = scale.solve(cam, solve_lines, {"facade": front, "points": lines["eave_left"][0], "length_in": 14 * FT})
    assert got.lines["drawn"].length_in == pytest.approx(true_length((40, 10, 0), (25, 17.5, 15)), rel=0.02)


def test_the_missing_slope_of_a_gable_is_a_rake():
    shot = Shot()
    cam, front, side = facade_names(shot)
    lines = [([shot.pixel((40, 10, 0)), shot.pixel((40, 17.5, 15))], "rake", None),
             ([shot.pixel((0, 10, 0)), shot.pixel((40, 10, 0))], "eave", front)]
    drawn_line = drawn.snap([_nudge(shot.pixel((40, 17.5, 15))), _nudge(shot.pixel((40, 10, 30)))],
                            [points for points, _, _ in lines], SIZE)
    assert drawn.suggest(drawn_line, lines, cam, SIZE)[0] == "rake"


def test_an_end_far_from_every_line_stays_where_it_was_drawn():
    shot, cam, front, lines = _house()
    far = [[100.0, 100.0], [300.0, 120.0]]
    assert drawn.snap(far, [points for points, _, _ in lines.values()], SIZE) == far
