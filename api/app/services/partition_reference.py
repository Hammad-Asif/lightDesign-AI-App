"""Partition mode, run by the reference implementation itself.

The product's own port of the method (services/partition.py) can never be pixel-identical to
the reference project it was ported from. When `partition_engine` is "reference", the roof
lines come from the reference package (`roofline`, installed from
/Users/samibutt/Documents/freelance/roof-detection), fed exactly what the reference app feeds it:
a working copy made with OpenCV INTER_AREA at a long edge of 2000, the SAM 3 instances that
came back for that copy, and the reference's own foliage and vanishing-point steps. Nothing
runs on top afterwards except the instruction's side/edge filter in the router.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from app.services import roofline as product_roofline

if TYPE_CHECKING:
    from roofline.config import Params

WORKING_LONG_EDGE = 2000

# The reference package comes from the separate roof-detection project, not from this repo, so
# it is loaded on first use rather than at import time: a checkout without it still starts, and
# every other mode still runs. Only PARTITION_ENGINE=reference needs it.
MISSING = (
    "PARTITION_ENGINE=reference needs the roof-detection project installed as the `roofline` "
    "package (pip install -e <path to roof-detection>). Without it here, set PARTITION_ENGINE=port "
    "to use this repo's own implementation of the same method, or EDGE_EXTRACTOR=hybrid."
)
_engine_cache: tuple | None = None


def available() -> bool:
    """Whether the reference package can be imported on this machine."""
    try:
        return importlib.util.find_spec("roofline") is not None
    except (ImportError, ValueError):
        return False


def _engine():
    """(partition, pipeline, Params) from the reference package, or a RuntimeError naming the fix."""
    global _engine_cache
    if _engine_cache is None:
        try:
            package = importlib.import_module("roofline")
            _engine_cache = (
                importlib.import_module("roofline.partition"),
                importlib.import_module("roofline.pipeline"),
                importlib.import_module("roofline.config").Params,
            )
            del package
        except ImportError as exc:
            raise RuntimeError(MISSING) from exc
    return _engine_cache


def working_copy(image: Image.Image) -> tuple[np.ndarray, Image.Image, float]:
    """(BGR array, PIL image, full/working scale) built the way the reference app builds it."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = WORKING_LONG_EDGE / max(h, w)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr, Image.fromarray(rgb), w / bgr.shape[1]


def _pitch_entry(plane: dict) -> dict | None:
    """The product's pitch shape, from the reference's per-plane fields."""
    out: dict = {}
    if plane.get("pitch_apparent_deg") is not None:
        deg = float(plane["pitch_apparent_deg"])
        out["as_seen"] = {"run": 12, "rise": round(math.tan(math.radians(min(deg, 80.0))) * 12, 1), "degrees": round(deg, 1)}
        std = (plane.get("pitch_apparent_standard") or {}).get("pitch")
        if std:
            out["nearest_standard"] = {"run": 12, "rise": int(std.split(":")[0]), "basis": "as_seen"}
    cal = plane.get("pitch_calibrated")
    if cal:
        out["calibrated"] = {"run": 12, "rise": round(math.tan(math.radians(cal["pitch_deg"])) * 12, 1), "degrees": round(cal["pitch_deg"], 1),
                             "edges_used": cal.get("edges_used")}
        std = (cal.get("standard") or {}).get("pitch")
        if std:
            out["nearest_standard"] = {"run": 12, "rise": int(std.split(":")[0]), "basis": "calibrated"}
    return out or None


def result(working_bgr: np.ndarray, working_masks: list[np.ndarray], scores: list[float | None], full_masks: list[np.ndarray],
           full_width: int, scale: float, foliage: np.ndarray | None = None, focal_35mm: float | None = None,
           params: Params | None = None) -> tuple[list[list[tuple]], set, dict, dict]:
    """The lines per plane as (points, side, edge) in full-resolution pixels, the keys of the
    inferred connectors, the pitch per plane, and the reference's own result for inspection."""
    partition, pipeline, Params = _engine()
    params = params or Params.load()
    instances = [{"mask": (m > 0).astype(np.uint8) * 255, "score": s, "source": "fal"} for m, s in zip(working_masks, scores)]
    prep = pipeline.prepare(working_bgr, instances, params, foliage_override=foliage)
    focal_px = focal_35mm / 36.0 * max(working_bgr.shape[:2]) if focal_35mm else None
    solved = pipeline.solve(prep, params, 7, focal_px, "EXIF" if focal_px else None)

    label_to_instance = {p["label"]: p["instance"] for p in prep.partition.planes}
    sides = [product_roofline.plane_side(mask, full_width) for mask in full_masks]
    planes: list[list[tuple]] = [[] for _ in full_masks]
    inferred: set = set()
    for seg in solved["segments"]:
        owners = [label_to_instance[p] for p in seg.get("planes", []) if p in label_to_instance]
        if not owners:
            continue
        owner = owners[0]
        (x1, y1), (x2, y2) = seg["points"]
        points = [[int(round(x1 * scale)), int(round(y1 * scale))], [int(round(x2 * scale)), int(round(y2 * scale))]]
        if points[0] == points[1]:
            continue
        if points[0][0] > points[1][0]:
            points = points[::-1]
        side = sides[owner]
        if len(owners) == 2 and sides[owners[0]] != sides[owners[1]]:
            side = "front" if "front" in (sides[owners[0]], sides[owners[1]]) else side
        planes[owner].append((points, side, seg["role"]))
        if seg.get("inferred"):
            inferred.add(tuple(map(tuple, points)))

    colors = partition.label_colors(max(len(prep.partition.planes), 1))
    solved["plane_polygons"] = [
        {"plane": int(label_to_instance.get(plane["plane"], -1)),
         "polygon": [[int(round(x * scale)), int(round(y * scale))] for x, y in plane["polygon"]],
         "color": "#%02x%02x%02x" % colors[(plane["plane"] - 1) % len(colors)][::-1]}
        for plane in solved["planes"]
    ]
    pitches: dict = {}
    for plane in solved["planes"]:
        instance = label_to_instance.get(plane["plane"])
        entry = _pitch_entry(plane)
        if instance is not None and entry and instance not in pitches:
            pitches[instance] = entry
    return planes, inferred, pitches, solved
