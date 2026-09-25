"""The two metrics every change is judged by.

1. Chamfer distance (both directions) between predicted and ground-truth edge pixels: one robust
   scalar, good for CI. Reported in pixels and as a percentage of the image diagonal.
2. Matching F1, in the style of structural AP for wireframes: a predicted segment is a true
   positive when both its endpoints lie within a tolerance of a ground-truth segment's endpoints
   (either orientation, one-to-one). Reported at several tolerances together with the mean
   angular error of the matches and the number of unmatched predictions, so over-segmentation
   shows up as a number and not just as clutter.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

TOLERANCES_FRAC = (0.005, 0.01, 0.02)     # of the image diagonal


def annotation_segments(annotation: dict, dedupe_tol: float = 4.0) -> np.ndarray:
    """Unique ground-truth segments (M, 4) from closed plane polygons plus any free lines.

    An edge shared by two planes appears once. `skip_edges` on a plane lists edge indices
    (edge k runs from vertex k to k+1) the annotator did not actually draw, e.g. behind a tree.
    """
    raw: list[tuple[float, float, float, float]] = []
    for plane in annotation.get("planes", []):
        poly = plane["polygon"]
        skip = set(plane.get("skip_edges", []))
        for k in range(len(poly)):
            if k in skip:
                continue
            (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % len(poly)]
            raw.append((x1, y1, x2, y2))
    for (x1, y1), (x2, y2) in annotation.get("lines", []):
        raw.append((x1, y1, x2, y2))

    unique: list[tuple[float, float, float, float]] = []
    for seg in raw:
        a, b = np.array(seg[:2]), np.array(seg[2:])
        if np.linalg.norm(a - b) < 1.0:
            continue
        duplicate = False
        for other in unique:
            c, d = np.array(other[:2]), np.array(other[2:])
            if max(np.linalg.norm(a - c), np.linalg.norm(b - d)) <= dedupe_tol or max(np.linalg.norm(a - d), np.linalg.norm(b - c)) <= dedupe_tol:
                duplicate = True
                break
        if not duplicate:
            unique.append(seg)
    return np.array(unique, float).reshape(-1, 4)


def _rasterize(segments: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in segments:
        cv2.line(canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), 255, 1)
    return canvas


def chamfer(pred: np.ndarray, gt: np.ndarray, shape: tuple[int, int]) -> dict:
    diag = math.hypot(*shape)
    if len(pred) == 0 or len(gt) == 0:
        return {"pred_to_gt_px": diag, "gt_to_pred_px": diag, "chamfer_px": diag, "chamfer_pct": 100.0}
    pred_px, gt_px = _rasterize(pred, shape), _rasterize(gt, shape)
    dist_to_gt = cv2.distanceTransform(255 - gt_px, cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform(255 - pred_px, cv2.DIST_L2, 5)
    p2g = float(dist_to_gt[pred_px > 0].mean())
    g2p = float(dist_to_pred[gt_px > 0].mean())
    total = (p2g + g2p) / 2.0
    return {"pred_to_gt_px": round(p2g, 2), "gt_to_pred_px": round(g2p, 2), "chamfer_px": round(total, 2),
            "chamfer_pct": round(total / diag * 100, 3)}


def _endpoint_distance(p: np.ndarray, g: np.ndarray) -> float:
    a = max(math.hypot(p[0] - g[0], p[1] - g[1]), math.hypot(p[2] - g[2], p[3] - g[3]))
    b = max(math.hypot(p[0] - g[2], p[1] - g[3]), math.hypot(p[2] - g[0], p[3] - g[1]))
    return min(a, b)


def _angle(seg: np.ndarray) -> float:
    return math.degrees(math.atan2(seg[3] - seg[1], seg[2] - seg[0])) % 180.0


def match_f1(pred: np.ndarray, gt: np.ndarray, tol_px: float) -> dict:
    pairs = sorted(((_endpoint_distance(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), key=lambda t: t[0])
    used_p, used_g, angular = set(), set(), []
    for d, i, j in pairs:
        if d > tol_px:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        diff = abs(_angle(pred[i]) - _angle(gt[j])) % 180.0
        angular.append(min(diff, 180.0 - diff))
    tp = len(used_p)
    precision = tp / len(pred) if len(pred) else 0.0
    recall = tp / len(gt) if len(gt) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tol_px": round(tol_px, 1), "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "mean_angular_error_deg": round(float(np.mean(angular)), 2) if angular else None,
            "unmatched_predictions": len(pred) - tp}


def evaluate(pred_segments, annotation: dict, shape: tuple[int, int]) -> dict:
    pred = np.array(pred_segments, float).reshape(-1, 4)
    gt = annotation_segments(annotation)
    diag = math.hypot(*shape)
    result = {"n_pred": int(len(pred)), "n_gt": int(len(gt)), **chamfer(pred, gt, shape)}
    for frac in TOLERANCES_FRAC:
        result[f"f1@{frac * 100:g}%"] = match_f1(pred, gt, frac * diag)
    return result


def score(result: dict) -> float:
    """One number to minimise in the outer loop: chamfer (% of diagonal) plus the F1 shortfall."""
    f1 = np.mean([result[f"f1@{frac * 100:g}%"]["f1"] for frac in TOLERANCES_FRAC])
    return float(result["chamfer_pct"] + 2.0 * (1.0 - f1))
