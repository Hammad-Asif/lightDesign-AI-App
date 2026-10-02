"""The V3 plane filter (services/fusion.plane_filter) on drawn masks: which lines are a
neighbour's. A neighbour the photo's edge cuts off goes even when it stands within the house's
columns, behind its far corner; an upper storey within them stays."""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import fusion
from app.services.rooflines import RoofLine

W, H = 1600, 1000


def _ctx(masks, keep):
    return fusion.Context(Image.new("RGB", (W, H)), None, masks, keep, None)


def _polygon(*points):
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
    return mask.astype(bool)


def _box(x0, y0, x1, y1):
    mask = np.zeros((H, W), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_a_line_on_a_neighbour_the_frame_cuts_is_rejected_inside_the_house_columns():
    # The target's hip roof (client test photo): eave 270-1490, ridge 515-1205, hip down to the
    # right corner. The neighbour's roof shows above that hip and runs off the right edge.
    house = _polygon((270, 435), (515, 270), (1205, 280), (1490, 440))
    neighbour = _polygon((1290, 300), (1330, 280), (W, 330), (W, 420), (1420, 390))
    ctx = _ctx([house, neighbour], [True, False])
    line = RoofLine([[1330.0, 285.0], [1555.0, 375.0]], ["v3"], "rake")   # mostly within the house's columns
    assert not fusion.plane_filter(line, ctx)
    assert line.plane_status == "rejected:cut_neighbour_plane"


def test_a_line_on_an_excluded_plane_the_frame_does_not_cut_keeps_the_upper_storey_rule():
    house = _box(270, 500, 1500, 700)
    upper = _box(600, 200, 1100, 330)          # excluded by the selection, wholly in the photo
    ctx = _ctx([house, upper], [True, False])
    line = RoofLine([[620.0, 210.0], [1080.0, 210.0]], ["v3"], "ridge")
    assert fusion.plane_filter(line, ctx)
    assert line.plane_status == "kept:within_house_width"
