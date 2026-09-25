"""Phase 2: clean each plane inside the partition, then split its boundary into arcs.

Cleanup never lets a plane grow over another plane: new pixels are only taken from background,
so the partition stays hard. The output is, per plane, an ordered boundary whose pixels carry
  * a tag: 0 = free (the other side is not roof) or j = adjacent to plane j, and
  * a trust flag: False where foliage stands in front of the roof or the image frame cuts it.
Untrusted pixels never take part in a line fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import Params
from .partition import Partition


@dataclass
class Boundary:
    plane: int                 # label in the partition
    part: int                  # a plane can have more than one blob
    contour: np.ndarray        # (N, 2) int, ordered, closed
    tags: np.ndarray           # (N,) int32: 0 free, j adjacent to plane j
    trusted: np.ndarray        # (N,) bool
    perimeter: float


def foliage_mask(image_bgr: np.ndarray, params: Params) -> np.ndarray:
    """Vegetation from excess green, dilated so it covers the mask edge it chewed."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(image_bgr)]
    exg = (2 * g - r - b) / (r + g + b + 1.0)
    veg = ((exg > params.foliage_exg) & (g > 35)).astype(np.uint8)
    veg = cv2.morphologyEx(veg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    veg = cv2.morphologyEx(veg, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(veg, connectivity=8)
    keep = np.zeros_like(veg)
    min_area = 0.0005 * veg.size
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    k = int(params.foliage_dilate_frac * max(veg.shape)) * 2 + 1
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def dilate_mask(mask: np.ndarray, params: Params) -> np.ndarray:
    """A foliage mask from elsewhere (a SAM "tree" pass), grown like the colour-based one."""
    k = int(params.foliage_dilate_frac * max(mask.shape)) * 2 + 1
    return cv2.dilate((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def _fill_narrow_notches(target: np.ndarray, contour: np.ndarray, params: Params, long_edge: int) -> None:
    """Fill bites out of a plane's edge: convexity defects that are deep, narrow-mouthed and small."""
    if len(contour) < 4:
        return
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return
    if defects is None:
        return
    area = cv2.contourArea(contour)
    for start, end, _, depth in defects.reshape(-1, 4):
        depth_px = depth / 256.0
        if depth_px < params.notch_min_depth_frac * long_edge:
            continue
        mouth = float(np.linalg.norm(contour[start, 0] - contour[end, 0]))
        if mouth > params.notch_max_mouth_ratio * depth_px:
            continue
        run = contour[start:end + 1, 0] if end >= start else np.concatenate([contour[start:, 0], contour[:end + 1, 0]])
        if len(run) < 3 or cv2.contourArea(run.astype(np.int32).reshape(-1, 1, 2)) > params.notch_max_area_frac * area:
            continue
        cv2.fillPoly(target, [run.astype(np.int32).reshape(-1, 1, 2)], 1)


def clean_partition(part: Partition, params: Params) -> np.ndarray:
    """Returns the cleaned label map (same labels as `part`; vanished planes simply have no pixels)."""
    labels = part.labels.copy()
    H, W = labels.shape
    long_edge = max(H, W)
    k = max(3, int(params.close_frac * long_edge) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    pad = k + 2
    for plane in part.planes:
        lab = plane["label"]
        ys, xs = np.nonzero(labels == lab)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W), max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
        view = labels[y0:y1, x0:x1]
        region = (view == lab).astype(np.uint8)
        background = view == 0

        grown = cv2.morphologyEx(region, cv2.MORPH_CLOSE, close_kernel)
        region = np.where(background & (grown > 0), 1, region).astype(np.uint8)
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            view[view == lab] = 0
            continue
        areas = [cv2.contourArea(c) for c in contours]
        keep = [c for c, a in zip(contours, areas) if a >= params.min_blob_frac * max(areas)]
        solid = np.zeros_like(region)
        cv2.drawContours(solid, keep, -1, 1, thickness=cv2.FILLED)       # fills holes...
        for c in keep:
            _fill_narrow_notches(solid, c, params, long_edge)
        own_or_free = (view == lab) | background                           # ...but never over another plane
        final = (solid > 0) & own_or_free
        if final.sum() < params.min_instance_frac * H * W:
            final[:] = False
        view[view == lab] = 0
        view[final] = lab
    return labels


def _smooth_tags(tags: np.ndarray, min_run: int) -> np.ndarray:
    """Runs shorter than `min_run` take the tag of the longer neighbouring run (cyclic)."""
    n = len(tags)
    if n < 2 * min_run:
        return tags
    out = tags.copy()
    for _ in range(3):
        change = np.nonzero(out != np.roll(out, 1))[0]
        if len(change) < 2:
            break
        starts = list(change)
        modified = False
        for a, b in zip(starts, starts[1:] + [starts[0] + n]):
            if b - a < min_run:
                idx = np.arange(a, b) % n
                out[idx] = out[(a - 1) % n]
                modified = True
        if not modified:
            break
    return out


def _rms_to_line(points: np.ndarray) -> float:
    centred = points - points.mean(axis=0)
    values = np.linalg.eigvalsh(centred.T @ centred / len(points))
    return float(np.sqrt(max(values[0], 0.0)))


def _split_straight(points: np.ndarray, rms_tol: float, min_len: float, out: list) -> None:
    """Recursive split at the point farthest from the chord until every piece is straight or short."""
    if len(points) < 2 * min_len and _rms_to_line(points) > rms_tol:
        out.append((len(points), False))                      # rough and too short to split again
        return
    if _rms_to_line(points) <= rms_tol:
        out.append((len(points), len(points) >= min_len))     # straight: trusted if it is also long
        return
    chord = points[-1] - points[0]
    norm = np.linalg.norm(chord)
    deviation = np.abs((points - points[0]) @ (np.array([-chord[1], chord[0]]) / norm)) if norm > 1e-9 else np.linalg.norm(points - points[0], axis=1)
    k = int(np.clip(np.argmax(deviation), 3, len(points) - 4))
    _split_straight(points[:k + 1], rms_tol, min_len, out)
    _split_straight(points[k:], rms_tol, min_len, out)


def _occluded(points: np.ndarray, suspect: np.ndarray, rms_tol: float, min_len: float) -> np.ndarray:
    """Per boundary pixel: True where a foliage-adjacent stretch is NOT made of long straight pieces."""
    n = len(points)
    flags = suspect.copy()
    change = np.nonzero(flags != np.roll(flags, 1))[0]
    if len(change) == 0:
        stretches = [np.arange(n)] if flags[0] else []
    else:
        runs = [(int(a), int(b)) for a, b in zip(change, list(change[1:]) + [int(change[0]) + n])]
        for a, b in runs:                                     # bridge pinholes in the foliage mask
            if not flags[a % n] and b - a <= 6:
                flags[np.arange(a, b) % n] = True
        change = np.nonzero(flags != np.roll(flags, 1))[0]
        if len(change) == 0:
            stretches = [np.arange(n)] if flags[0] else []
        else:
            runs = [(int(a), int(b)) for a, b in zip(change, list(change[1:]) + [int(change[0]) + n])]
            stretches = [np.arange(a, b) % n for a, b in runs if flags[a % n] and b - a >= 8]
    occluded = np.zeros(n, bool)
    for idx in stretches:
        pieces: list[tuple[int, bool]] = []
        _split_straight(points[idx], rms_tol, min_len, pieces)
        cursor = 0
        for length, ok in pieces:
            if not ok:
                occluded[idx[cursor:cursor + length]] = True
            cursor += max(length - 1, 1)                      # pieces share their split point
    return occluded


def extract_boundaries(labels: np.ndarray, params: Params, foliage: Optional[np.ndarray] = None,
                       untrusted: Optional[np.ndarray] = None) -> list[Boundary]:
    """Ordered plane boundaries with per-pixel tags and trust.

    Foliage next to an edge is not enough to distrust it: trees behind a ridge or lawn under an
    eave border perfectly good edges. What gives an occluder away is its outline: a roof edge is
    straight, a tree is not. Each foliage-adjacent stretch of boundary is split recursively into
    straight pieces; pieces that are long and straight stay trusted, short or rough ones are the
    outline of something standing in front of the roof. `untrusted` is an optional mask that
    removes trust unconditionally.
    """
    H, W = labels.shape
    reach = params.adjacency_reach_px
    kernel = np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)
    boundaries: list[Boundary] = []
    for lab in [int(v) for v in np.unique(labels) if v > 0]:
        ys, xs = np.nonzero(labels == lab)
        pad = reach + 2
        x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W), max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
        view = labels[y0:y1, x0:x1]
        region = (view == lab).astype(np.uint8)
        others = np.where(view == lab, 0, view).astype(np.uint16)
        near = cv2.dilate(others, kernel)                                  # label of a plane within reach, else 0
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted([c for c in contours if len(c) >= 12], key=cv2.contourArea, reverse=True)
        for part, c in enumerate(contours):
            pts = c.reshape(-1, 2)
            tags = _smooth_tags(near[pts[:, 1], pts[:, 0]].astype(np.int32), min_run=6)
            gx, gy = pts[:, 0] + x0, pts[:, 1] + y0
            trusted = (gx > 1) & (gy > 1) & (gx < W - 2) & (gy < H - 2)
            if foliage is not None and foliage[gy, gx].any():
                min_len = max(30.0, params.straight_min_len_frac * float(cv2.arcLength(c, True)))
                trusted &= ~_occluded(pts.astype(float), foliage[gy, gx], params.straight_rms_px, min_len)
            if untrusted is not None:
                trusted &= ~untrusted[gy, gx]
            boundaries.append(Boundary(lab, part, np.stack([gx, gy], axis=1).astype(np.int32), tags, trusted,
                                       float(cv2.arcLength(c, True))))
    return boundaries


def render_arcs(image_bgr: np.ndarray, boundaries: list[Boundary], foliage: Optional[np.ndarray]) -> np.ndarray:
    """Inspection view: free arcs green, adjacency arcs yellow, untrusted pixels red, foliage tinted."""
    vis = (image_bgr * 0.55).astype(np.uint8)
    if foliage is not None:
        vis[foliage] = (vis[foliage] * 0.5 + np.array([60, 20, 90]) * 0.5).astype(np.uint8)
    for b in boundaries:
        colors = np.where(b.tags[:, None] > 0, np.array([[0, 230, 255]]), np.array([[60, 255, 60]]))
        colors = np.where(b.trusted[:, None], colors, np.array([[40, 40, 255]]))
        for (x, y), col in zip(b.contour, colors):
            cv2.circle(vis, (int(x), int(y)), 1, tuple(int(v) for v in col), -1)
    return vis
