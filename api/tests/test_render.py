"""M3 Day 5: the whole chain on a rendered photo. A corner house is drawn with Pillow by a known
pinhole camera; the app's own vanishing-point detection reads the image, services/camera.py builds
the camera from it, and services/scale.py measures the house from one 16 ft reference.

Unlike test_scale.py, nothing here is handed the true vanishing points. Tolerances are what the
detector achieves on a clean image, measured over these poses: focal within 3%, the measured wall
within 1%, walls and rakes that inherit their scale through a corner within 5%. Real photos add
their own error on top; Lukas's acceptance (10% of tape on 8 of 10 houses) is judged on those,
with tests/measure_eval.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.config import Settings
from app.services import camera as camera_module
from app.services import fusion, scale
from test_scale import FT, H, W, F, Shot


def render(shot: Shot) -> Image.Image:
    """A 40 x 30 ft house, 10 ft walls with siding every 6 in, a 6/12 side-gable roof; the camera
    looks at its front-right corner."""
    image = Image.new("RGB", (W, H), (150, 185, 225))
    draw = ImageDraw.Draw(image)
    px = shot.pixel
    draw.rectangle([0, H * 0.72, W, H], fill=(90, 140, 80))
    draw.polygon([tuple(px(p)) for p in [(0, 0, 0), (40, 0, 0), (40, 10, 0), (0, 10, 0)]], fill=(225, 214, 190))
    draw.polygon([tuple(px(p)) for p in [(40, 0, 0), (40, 0, 30), (40, 10, 30), (40, 10, 0)]], fill=(200, 188, 165))
    for y in np.arange(0.5, 10, 0.5):
        draw.line([tuple(px((0, y, 0))), tuple(px((40, y, 0)))], fill=(120, 105, 85), width=2)
        draw.line([tuple(px((40, y, 0))), tuple(px((40, y, 30)))], fill=(110, 95, 78), width=2)
    for x in (0, 10, 20, 30, 40):
        draw.line([tuple(px((x, 0, 0))), tuple(px((x, 10, 0)))], fill=(100, 90, 75), width=3)
    for z in (10, 20, 30):
        draw.line([tuple(px((40, 0, z))), tuple(px((40, 10, z)))], fill=(100, 90, 75), width=3)
    draw.polygon([tuple(px(p)) for p in [(0, 10, 0), (40, 10, 0), (40, 17.5, 15), (0, 17.5, 15)]], fill=(95, 70, 55))
    draw.polygon([tuple(px(p)) for p in [(40, 10, 0), (40, 17.5, 15), (40, 10, 30)]], fill=(210, 198, 172))
    return image


@pytest.mark.parametrize("yaw, pitch", [(-10, 3), (-30, 6), (-40, 8)])
def test_a_rendered_corner_house_is_measured_from_one_reference(yaw, pitch):
    shot = Shot(yaw_deg=yaw, pitch_deg=pitch)
    image = render(shot)
    scene = fusion.Context(image, Settings(_env_file=None), [], None, None).scene
    cam = camera_module.build(scene, (W, H), None)
    assert cam.focal_source == "vp" and cam.focal == pytest.approx(F, rel=0.03)
    assert len(cam.facades) == 2

    px = shot.pixel
    front = cam.facade_of(px((0, 10, 0)), px((40, 10, 0)), 3.0)
    assert front is not None
    garage_door = {"facade": front, "points": [px((4, 2, 0)), px((20, 2, 0))], "length_in": 16 * FT}
    gable = scale.solve(cam, [
        scale.Line("eave", [px((0, 10, 0)), px((40, 10, 0))], "eave", front),
        scale.Line("rake1", [px((40, 10, 0)), px((40, 17.5, 15))], "rake", None),
        scale.Line("rake2", [px((40, 17.5, 15)), px((40, 10, 30))], "rake", None),
    ], garage_door).lines
    assert gable["eave"].length_in == pytest.approx(40 * FT, rel=0.01)
    for rake in ("rake1", "rake2"):
        assert gable[rake].length_in == pytest.approx(np.hypot(15, 7.5) * FT, rel=0.05)
    # A hip house's side eave: detection gave it no side; its vanishing point does.
    hip = scale.solve(cam, [
        scale.Line("eave", [px((0, 10, 0)), px((40, 10, 0))], "eave", front),
        scale.Line("side", [px((40, 10, 0)), px((40, 10, 30))], "eave", None),
    ], garage_door).lines
    assert hip["side"].length_in == pytest.approx(30 * FT, rel=0.05)


def test_a_vertical_vanishing_point_that_is_not_vertical_is_refitted_from_upright_segments():
    shot = Shot(yaw_deg=-30, pitch_deg=6)
    scene = fusion.Context(render(shot), Settings(_env_file=None), [], None, None).scene
    true_vertical = camera_module.Camera(W, H, F, "vp").K @ (shot.R @ np.array([0.0, 1.0, 0.0]))
    cam = camera_module.build(scene, (W, H), None)
    # Detection on this image takes the steep side-wall siding for the vertical; the camera must not.
    assert "vertical_rejected" in cam.focal_checks
    found = np.array(cam.vertical[:2]) / cam.vertical[2]
    truth = true_vertical[:2] / true_vertical[2]
    direction = lambda p: (p - np.array([W / 2, H / 2])) / np.linalg.norm(p - np.array([W / 2, H / 2]))  # noqa: E731
    assert float(direction(found) @ direction(truth)) > np.cos(np.radians(1.0))
