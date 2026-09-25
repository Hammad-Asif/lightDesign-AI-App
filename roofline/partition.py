"""Phase 1: turn overlapping SAM instances into a hard partition (one label per pixel).

SAM 3 returns one instance per roof plane, but instances overlap: a parent plane and its
sub-plane both come back, and neighbours bleed into each other. A hard partition makes the border
between two labels a well-defined curve, and that border is where ridges, hips and valleys live.

Priority: the smallest instance wins a contested pixel, so children beat parents. Score is folded
into the priority (area / score^gamma) and instances under the soft floor are pushed back, never
dropped outright. An instance that keeps less than `min_keep_fraction` of its own mask after the
contest is a duplicate of whatever beat it and is removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Params


@dataclass
class Partition:
    labels: np.ndarray                  # int32 (H, W); 0 = background, k >= 1 = plane k
    planes: list[dict]                  # one per label: {"label", "instance", "score", "area", "kept_fraction", "effective_size"}
    dropped: list[dict] = field(default_factory=list)


def _assign(masks: list[np.ndarray], order: list[int], shape: tuple[int, int]) -> np.ndarray:
    labels = np.zeros(shape, np.int32)
    for idx in order:
        free = masks[idx] & (labels == 0)
        labels[free] = idx + 1
    return labels


def close_gaps(labels: np.ndarray, kernel_px: int) -> np.ndarray:
    """Give narrow unlabelled gaps (between planes, or notches) to the nearest plane."""
    if kernel_px < 3 or not labels.any():
        return labels
    union = (labels > 0).astype(np.uint8)
    closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px)))
    gaps = (closed > 0) & (union == 0)
    if not gaps.any():
        return labels
    _, nearest = cv2.distanceTransformWithLabels(1 - union, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(nearest.max()) + 1, np.int32)
    lut[nearest[union > 0]] = labels[union > 0]
    out = labels.copy()
    out[gaps] = lut[nearest[gaps]]
    return out


def build_partition(instances: list[dict], shape: tuple[int, int], params: Params) -> Partition:
    masks = [inst["mask"] > 0 for inst in instances]
    areas = [int(m.sum()) for m in masks]
    scores = [1.0 if inst.get("score") is None else float(inst["score"]) for inst in instances]
    effective = [a / max(s, 0.05) ** params.score_gamma * (params.floor_penalty if s < params.score_floor else 1.0)
                 for a, s in zip(areas, scores)]

    long_edge = max(shape)
    rim = max(3, int(0.005 * long_edge) | 1)
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim, rim))
    min_area = params.min_instance_frac * shape[0] * shape[1]

    def solid_fraction(i: int, labels: np.ndarray) -> float:
        """Share of instance i that survives the contest as a real piece (thin rims are opened away)."""
        left = (labels == i + 1).astype(np.uint8)
        if left.sum() >= params.min_keep_fraction * areas[i]:
            return float(left.sum()) / areas[i]
        solid = int(cv2.morphologyEx(left, cv2.MORPH_OPEN, rim_kernel).sum())
        return solid / areas[i] if solid >= min_area else 0.0

    alive = [i for i, a in enumerate(areas) if a > 0]
    dropped: list[dict] = []
    while True:
        order = sorted(alive, key=lambda i: effective[i])
        labels = _assign(masks, order, shape)
        kept = {i: solid_fraction(i, labels) for i in alive}
        worst = min(alive, key=lambda i: kept[i], default=None)
        if worst is None or kept[worst] >= params.min_solid_fraction:
            break
        dropped.append({"instance": worst, "score": scores[worst], "kept_fraction": round(kept[worst], 3),
                        "reason": "duplicate: a higher-priority instance took its mask and left only a rim"})
        alive.remove(worst)

    labels = close_gaps(labels, int(params.gap_close_frac * long_edge) | 1)

    # compact relabel 1..K, largest plane first
    planes = []
    remap = np.zeros(len(instances) + 1, np.int32)
    for new, i in enumerate(sorted(alive, key=lambda i: -areas[i]), start=1):
        remap[i + 1] = new
        planes.append({"label": new, "instance": i, "score": scores[i] if instances[i].get("score") is not None else None,
                       "area": areas[i], "kept_fraction": round(kept[i], 3), "effective_size": round(effective[i], 1)})
    return Partition(remap[labels], planes, dropped)


def label_colors(n: int) -> list[tuple[int, int, int]]:
    """Distinct BGR colours (golden-angle hue spacing)."""
    out = []
    for i in range(n):
        hue = int((i * 137.508) % 360) // 2
        out.append(tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 190, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
    return out


def render_labels(labels: np.ndarray) -> np.ndarray:
    n = int(labels.max())
    lut = np.zeros((n + 1, 3), np.uint8)
    for i, c in enumerate(label_colors(n), start=1):
        lut[i] = c
    return lut[labels]
