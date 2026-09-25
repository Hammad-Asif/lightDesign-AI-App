"""Phase 6: creases that a single SAM mask hid.

Most interior lines already exist as borders of the partition. This phase only runs where one
instance genuinely spans two roof faces, and LSD is never an output: it can only CONFIRM an edge
that was hypothesised from the geometry. A hypothesis is a chord between two non-adjacent corners
of one plane, lying inside it. It is kept only if it is long, is not parallel to the plane's eave
(tile rows would fake the support), and is covered along most of its length by LSD segments that
agree with it in direction and position. The confirmed chord splits the plane in two.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .config import Params
from .vanishing import VanishingPoints


def _coverage(a: np.ndarray, b: np.ndarray, lsd: np.ndarray, perp_px: float, angle_deg: float) -> float:
    """Share of chord a-b covered by agreeing LSD segments (union of their projections)."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6 or len(lsd) == 0:
        return 0.0
    d /= length
    n = np.array([-d[1], d[0]])
    p1, p2 = lsd[:, :2] - a, lsd[:, 2:] - a
    sd = p2 - p1
    sl = np.maximum(np.linalg.norm(sd, axis=1), 1e-9)
    aligned = np.abs((sd @ d) / sl) >= math.cos(math.radians(angle_deg))
    close = (np.abs(p1 @ n) <= perp_px) & (np.abs(p2 @ n) <= perp_px)
    use = aligned & close
    if not use.any():
        return 0.0
    t1, t2 = p1[use] @ d, p2[use] @ d
    spans = sorted((max(0.0, min(x, y)), min(length, max(x, y))) for x, y in zip(t1, t2))
    covered, cursor = 0.0, 0.0
    for lo, hi in spans:
        if hi <= cursor:
            continue
        covered += hi - max(lo, cursor)
        cursor = hi
    return covered / length


def find_creases(planes: list[dict], segments: list[dict], lsd: np.ndarray, vps: Optional[VanishingPoints],
                 shape: tuple[int, int], params: Params) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (planes, segments, report). Planes that gain a crease are split in two."""
    diag = math.hypot(*shape)
    min_len = params.crease_min_len_frac * diag
    existing = {tuple(sorted(s["junctions"])) for s in segments if "junctions" in s}
    next_plane = max([p["plane"] for p in planes], default=0) + 1
    report, out_planes = [], []
    new_segments = list(segments)

    for plane in planes:
        ids, poly = plane["junctions"], np.array(plane["polygon"], float)
        m = len(ids)
        best = None
        if m >= 4 and len(lsd):
            # directions of this plane's own near-horizontal edges: tile rows run parallel to them
            eave_dirs = []
            for k in range(m):
                e = poly[(k + 1) % m] - poly[k]
                ln = np.linalg.norm(e)
                if ln < 1e-6:
                    continue
                e = e / ln
                horizontal = abs(math.degrees(math.atan2(e[1], e[0])) % 180 - 90) > 78
                if vps is not None and vps.horizontals:
                    mid = (poly[k] + poly[(k + 1) % m]) / 2
                    horizontal = any(abs(float(e @ d)) >= math.cos(math.radians(params.horizontal_vp_deg))
                                     for d in (vps.direction_at(h, mid) for h in vps.horizontals) if d is not None)
                if horizontal:
                    eave_dirs.append(e)
            contour = poly.astype(np.float32).reshape(-1, 1, 2)
            for i in range(m):
                for j in range(i + 2, m):
                    if i == 0 and j == m - 1:
                        continue                                           # adjacent around the cycle
                    if tuple(sorted((ids[i], ids[j]))) in existing:
                        continue
                    a, b = poly[i], poly[j]
                    chord = b - a
                    length = float(np.linalg.norm(chord))
                    if length < min_len:
                        continue
                    u = chord / length
                    if any(abs(float(u @ e)) >= math.cos(math.radians(params.crease_eave_parallel_deg)) for e in eave_dirs):
                        continue
                    if any(cv2.pointPolygonTest(contour, tuple(map(float, a + t * chord)), False) <= 0 for t in (0.25, 0.5, 0.75)):
                        continue
                    support = _coverage(a, b, lsd, params.crease_perp_px, params.crease_angle_deg)
                    if support >= params.crease_min_support and (best is None or support * length > best[0]):
                        best = (support * length, i, j, support)
        if best is None:
            out_planes.append(plane)
            continue
        _, i, j, support = best
        first = {"plane": plane["plane"], "part": plane["part"], "junctions": ids[i:j + 1], "polygon": plane["polygon"][i:j + 1]}
        second = {"plane": next_plane, "part": 0, "junctions": ids[j:] + ids[:i + 1], "polygon": plane["polygon"][j:] + plane["polygon"][:i + 1]}
        if cv2.contourArea(np.array(first["polygon"], np.float32)) < cv2.contourArea(np.array(second["polygon"], np.float32)):
            first["plane"], second["plane"] = second["plane"], first["plane"]
        out_planes += [first, second]
        new_segments.append({"points": [plane["polygon"][i], plane["polygon"][j]], "junctions": [ids[i], ids[j]],
                             "planes": sorted([first["plane"], second["plane"]]), "line": -1, "vp": None, "fitted": True,
                             "inferred": False, "crease": True, "lsd_support": round(support, 3)})
        # edges of the old plane now belong to whichever half holds both of their junctions
        second_ids = set(second["junctions"])
        for s in new_segments[:-1]:
            if plane["plane"] in s.get("planes", []) and "junctions" in s and set(s["junctions"]) <= second_ids and not set(s["junctions"]) <= set(first["junctions"]):
                s["planes"] = sorted({second["plane"] if p == plane["plane"] else p for p in s["planes"]})
        report.append({"split_plane": plane["plane"], "into": [first["plane"], second["plane"]], "lsd_support": round(support, 3),
                       "points": [plane["polygon"][i], plane["polygon"][j]]})
        next_plane += 1
    return out_planes, new_segments, report
