"""Pull an extracted line onto the image edge it is meant to follow.

SAM's mask boundary lands near the gutter but not always on it. The gutter, fascia and rake
edges are strong, straight gradients in the photo, so each point of a line is moved along the
line's own normal onto the strongest parallel edge within a short reach.

Three things matter, all measured on the annotated houses:

- Search **perpendicular to the line**, not straight down. On a 40-degree rake the offset to the
  true edge is mostly sideways, and a vertical search slides the point along the roof instead.
- **Smooth the offsets** before applying them. Per-point snapping bends a straight eave into a
  wobble that follows tile courses and shadows. It also flatters any score that measures
  distance to a line, which is how a wandering line can look more accurate than it is.
- **Only move on real evidence.** A point whose strongest neighbour is weaker than the image's
  own strong edges keeps its place.
"""

import cv2
import numpy as np

BAND = 15.0            # px: how far either side of the line to look
STEP = 2.0             # px between samples along the line
SMOOTH = 15            # samples in the median filter (about 30px of line)
STRENGTH_PERCENTILE = 90   # an edge must be as strong as the image's strongest tenth
MIN_STRENGTH = 8.0     # ... and this strong outright, so a flat image moves nothing
NEARER = 1e-3          # tie-break: with two equal edges, take the closer one
SIMPLIFY = 2.0         # px: tolerance when the snapped line is simplified back to a polyline
MIN_POINTS = 4         # shorter runs have nothing to smooth, so they are left alone
FIT_TOLERANCE = 2.0    # px a snapped sample may sit off the fitted line and still count
FIT_ROUNDS = 200       # random pairs tried when fitting
CONSENSUS = 0.5        # share of samples that must agree before the fit replaces the run


def field(image) -> tuple[np.ndarray, np.ndarray, float]:
    """Gradients of the photo, and the strength an edge must reach before a point moves."""
    grey = cv2.GaussianBlur(np.asarray(image.convert("L"), dtype=np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, max(float(np.percentile(np.hypot(gx, gy), STRENGTH_PERCENTILE)), MIN_STRENGTH)


def _median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    half = min(window, len(values)) // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i : i + 2 * half + 1]) for i in range(len(values))])


def _offsets(points: np.ndarray, gradients, band: float, minimum: float):
    """For each sample along one straight run: the point, its normal, and where its edge lies."""
    gx, gy = gradients
    height, width = gx.shape
    a, b = points
    direction = b - a
    length = float(np.linalg.norm(direction))
    if length < STEP:
        return None
    along = direction / length
    normal = np.array([-along[1], along[0]])
    reach = np.arange(-band, band + 1)

    samples = a + np.outer(np.arange(0.0, length, STEP), along)
    xs = np.clip(np.rint(samples[:, 0, None] + normal[0] * reach).astype(int), 0, width - 1)
    ys = np.clip(np.rint(samples[:, 1, None] + normal[1] * reach).astype(int), 0, height - 1)
    across = np.abs(gx[ys, xs] * normal[0] + gy[ys, xs] * normal[1])
    best = np.argmax(across - NEARER * np.abs(reach), axis=1)
    offsets = np.where(across[np.arange(len(best)), best] >= minimum, reach[best], 0.0)
    return samples, normal, offsets


def _fit(samples: np.ndarray, ends: np.ndarray, threshold: float, rounds: int):
    """A straight line through the snapped samples, as the run's two ends. None if they disagree.

    A roof edge is straight, so the run that follows it should be too: fitting throws away the
    points that wandered onto a tile course or a shadow instead of the edge.
    """
    if len(samples) < MIN_POINTS + 2:
        return None
    seed = np.random.default_rng(0)
    best = None
    for _ in range(rounds):
        i, j = seed.choice(len(samples), 2, replace=False)
        direction = samples[j] - samples[i]
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        inliers = np.abs((samples - samples[i]) @ normal) <= threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.mean() < CONSENSUS:
        return None
    kept = samples[best]
    centre = kept.mean(axis=0)
    axis = np.linalg.svd(kept - centre, full_matrices=False)[2][0]
    return np.stack([centre + axis * float((end - centre) @ axis) for end in ends])


def line(points, gradients, minimum: float, band: float = BAND, smooth: int = SMOOTH,
         simplify: float = SIMPLIFY, fit: bool = True) -> list[list[int]]:
    """One polyline moved onto the image edges it follows, run by run.

    Each straight run of the polyline is treated on its own, so a line that legitimately turns —
    two rakes joined at a gable apex — keeps its corner instead of being fitted into one line.
    """
    return fitted_line(points, gradients, minimum, band, smooth, simplify, fit)[0]


def fitted_line(points, gradients, minimum: float, band: float = BAND, smooth: int = SMOOTH,
                simplify: float = SIMPLIFY, fit: bool = True) -> tuple[list[list[int]], bool]:
    """`line`, and whether every run that moved found a straight fit.

    False means at least one run kept its median-smoothed points because no straight line had
    the consensus — the wobble a caller with its own straight geometry may prefer to keep out.
    """
    original = np.asarray(points, dtype=float)
    if len(original) < 2:
        return [[int(x), int(y)] for x, y in original], True

    straight = True
    runs: list[np.ndarray] = []
    for pair in zip(original[:-1], original[1:]):
        ends = np.stack(pair)
        found = _offsets(ends, gradients, band, minimum)
        if found is None:
            runs.append(ends)
            continue
        samples, normal, offsets = found
        if len(samples) < MIN_POINTS:
            runs.append(ends)
            continue
        moved = samples + normal * _median(offsets, smooth)[:, None]
        fitted = _fit(moved, ends, FIT_TOLERANCE, FIT_ROUNDS) if fit else None
        # Without a straight answer the run keeps the smoothed points it found.
        if fit and fitted is None:
            straight = False
        runs.append(fitted if fitted is not None else moved)

    # Where two runs meet, both moved their shared corner; sit it between the two answers.
    joined: list[np.ndarray] = [runs[0]]
    for previous, current in zip(runs, runs[1:]):
        corner = (previous[-1] + current[0]) / 2
        joined[-1] = np.vstack([joined[-1][:-1], corner])
        joined.append(np.vstack([corner, current[1:]]))
    run = np.vstack([piece if i == 0 else piece[1:] for i, piece in enumerate(joined)])
    if simplify > 0 and len(run) > 2:
        run = cv2.approxPolyDP(run.astype(np.float32).reshape(-1, 1, 2), simplify, False).reshape(-1, 2)
    return [[int(round(x)), int(round(y))] for x, y in run], straight


def lines(spans: list[tuple], gradients, minimum: float, **options) -> list[tuple]:
    """Every (points, side, edge) span, snapped. Side and edge are decided from the mask and
    never change here: this moves a line, it does not reinterpret it."""
    return [(line(points, gradients, minimum, **options), side, edge) for points, side, edge in spans]
