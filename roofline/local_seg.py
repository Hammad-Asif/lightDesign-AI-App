"""Segmenters that need no API key: point-seeded GrabCut and a plain threshold. They produce the same
instance dicts as SAM 3, one per click, so the rest of the pipeline is identical."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

GRABCUT_MAX_DIM = 720
GRABCUT_ITERATIONS = 5


def keep_component_at(mask: np.ndarray, point: Optional[tuple[int, int]]) -> np.ndarray:
    """Keep the connected component containing `point`, or the largest one if that misses."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    label = 0
    if point is not None:
        x, y = point
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            label = int(labels[y, x])
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (windows, texture) so each blob is a single clean silhouette."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def segment_mock(image_bgr: np.ndarray, point: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Local stand-in for a SAM point-prompt call, using GrabCut initialised from the point:
      * a thin band along the image border    -> definite background
      * everything else                       -> probable background
      * a box around the prompt               -> probable foreground (generous for the image
                                                 centre, tighter around an explicit click so
                                                 one click means one roof section)
      * a small disc around the prompt        -> definite foreground
    Nothing is asserted from colour (no "sky" rule): pale walls and overcast skies are too
    similar for that to be safe, and GrabCut's own colour models handle it well enough.
    Returns a uint8 mask (0/255) at the input resolution.
    """
    H, W = image_bgr.shape[:2]
    scale = min(1.0, GRABCUT_MAX_DIM / max(H, W))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    h, w = small.shape[:2]

    if point is None:
        px, py = w // 2, int(h * 0.45)
    else:
        px = int(np.clip(point[0] * scale, 0, w - 1))
        py = int(np.clip(point[1] * scale, 0, h - 1))

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    # Probable foreground box around the prompt, biased upward (roofs sit above the click).
    bw, bh_up, bh_down = (0.42, 0.42, 0.36) if point is None else (0.2, 0.16, 0.12)
    x0, x1 = max(0, px - int(w * bw)), min(w, px + int(w * bw))
    y0, y1 = max(0, py - int(h * bh_up)), min(h, py + int(h * bh_down))
    gc[y0:y1, x0:x1] = cv2.GC_PR_FGD

    border = max(2, int(min(h, w) * 0.02))
    gc[:border, :] = cv2.GC_BGD
    gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = cv2.GC_BGD
    gc[:, -border:] = cv2.GC_BGD

    cv2.circle(gc, (px, py), max(4, int(min(h, w) * 0.03)), int(cv2.GC_FGD), thickness=cv2.FILLED)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc, None, bgd_model, fgd_model, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Degenerate init (e.g. no background pixels left). Fall back to the plain threshold.
        return segment_threshold(image_bgr, 80, invert=True)

    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.012) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # bridge small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # drop 1-2 px slivers
    mask = keep_component_at(mask, (px, py))
    mask = fill_holes(mask)

    if scale < 1:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def segment_threshold(image_bgr: np.ndarray, threshold: int = 80, invert: bool = True) -> np.ndarray:
    """Deterministic fallback: blur -> grayscale threshold -> tidy -> largest blob -> fill holes.

    With `invert=True` (the default) dark pixels become foreground, which suits the common case
    of a dark roof against a bright sky.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, int(threshold), 255, mode)

    k = max(3, int(min(mask.shape[:2]) * 0.006) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = keep_component_at(mask, None)
    return fill_holes(mask)
