"""Roof lines from a hard partition of the plane masks (EDGE_EXTRACTOR=partition).

A port of the reference method ("How Roofline detects roof planes and turns them into lines"),
kept step for step so the two can be compared:

  2. hard partition   every pixel belongs to exactly one plane; smallest instance wins
  3. cleanup + trust  per-plane morphology, boundary tags, trusted arcs (frame, foliage)
  4. lines per plane  Douglas-Peucker seeds corners; Huber + total-least-squares fits;
                      corners are line intersections, carried straight through a gap
  5. vanishing points LSD on the whole photo, a vertical VP and a horizontal frame; each
                      roof edge near a VP direction is refit with that direction as a prior
  6. planar graph     shared lines merged across planes, junctions clustered, an energy
                      relaxed by block coordinate descent
  7. interior creases a chord inside a plane that LSD confirms splits the plane
  8. roles            ridge / hip / valley / eave / rake from topology

The one rule everything follows: the masks are plane support regions, not the source of
lines. The raw outline is never drawn and no line-detector output is ever drawn.

Sizes are fractions of the working image's long edge (L) or diagonal (D). The photo and masks
are worked at a long edge of at most WORK_LONG_EDGE and the lines scaled back.

What the product adds on top: the roof-to-wall junction test (roofline.mark_junctions), the
front/left/right side, and roof-relative snapping options. Step 1 (detection) is the
product's fal provider and is not changed here.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services import occlusion, roofline

WORK_LONG_EDGE = 2000

# Step 2
SCORE_FLOOR = 0.35            # below this an instance's effective size is quadrupled: it loses contests
SCORE_FLOOR_PENALTY = 4.0
KEEP_SHARE = 0.30             # an instance keeping this much of itself after the contest stays
RIM_OPEN_L = 0.005            # ellipse that removes thin rims from a smaller leftover
SOLID_SHARE = 0.10            # ... and the solid remainder must be this share of the instance
PLANE_MIN_IMAGE_SHARE = 0.0002
GAP_CLOSE_L = 0.006

# Step 3
CLEAN_CLOSE_L = 0.008
BLOB_MIN_SHARE = 0.05
NOTCH_DEPTH_L = 0.01
NOTCH_MOUTH_RATIO = 2.0
NOTCH_AREA_SHARE = 0.08
TAG_REACH = 3                 # px: another plane this close across a boundary pixel tags it
TAG_MIN_RUN = 6               # px: shorter tag runs are smoothed into their neighbours
FRAME_PX = 2
EXG_THRESHOLD = 0.06
FOLIAGE_MIN_SHARE = 0.0005
FOLIAGE_DILATE_L = 0.006
# A *detected* blocker (SAM's palm trunk, tree, car) is grown a little wider than the colour
# guess, so the boundary pixels the mask traced along its edge fall inside it.
BLOCKER_DILATE_L = 0.010
STRAIGHT_RMS = 3.0            # px: a suspect piece straighter than this is a real edge
TRUSTED_MIN_PX = 30
TRUSTED_MIN_PERIMETER = 0.04

# Step 4
SEED_EPS_LARGEST = 0.010      # of the largest plane's perimeter
SEED_EPS_OWN = 0.04           # of the plane's own perimeter
RUN_MIN_TRUSTED = 8
RUN_MIN_TRUSTED_SHARE = 0.35
FIT_TRIM_PX = 2.5
MERGE_DEG = 5.0
MERGE_MIN_EDGES = 3
CORNER_REACH_L = 0.02
GAP_REACH = 1.5

# Step 5
LSD_LONG_EDGE = 1200
LSD_MIN_D = 0.015
VP_AGREE_DEG = 2.0
VP_UPRIGHT_DEG = 20.0
VP_RANSAC_ROUNDS = 750
VP_CANDIDATES = 400
VP_FOCALS = 14
VP_TRUST_SHARE = 0.10
PRIOR_DEG = 4.0
PRIOR_LAMBDA = 3.0
# Fallback when no VP has enough support: the fitted edges' own 2-D directions are clustered
# (stiffness-weighted, greedy, 3 degrees on doubled angles) and each cluster mean is the prior
# at a lower weight. Off whenever a VP is usable: measured negative under perspective.
CLUSTER_DEG = 3.0
CLUSTER_LAMBDA = 2.0

# Step 6
LINE_MERGE_DEG = 5.0
LINE_MERGE_OFFSET_L = 0.006
LINE_MERGE_GAP_L = 0.25
# One physical edge drawn twice. The line merge in step 6 runs on the raw fits; the energy
# relaxation afterwards pulls two groups onto the same edge, so lines that were too far apart to
# merge can finish a pixel apart. Same test as that merge, applied to where the lines end up —
# and only between segments carrying the same role, because a thin band's eave and the junction
# above it are also near-parallel and close, and both of those are real.
DUPLICATE_DEG = 8.0
DUPLICATE_OFFSET_L = 0.006
DUPLICATE_OVERLAP = 0.70
JUNCTION_L = 0.008
# 0 = off, as in the reference. 10 kept thin tile bands' eave and ridge corners apart on the
# 1280px masks (house02 35.6 -> 49.5) but measured negative on the reference harness; to be
# re-measured once detection runs at the reference's 2000px / 12 masks.
JUNCTION_MIN_ANGLE_DEG = 0.0
JUNCTION_WEIGHT = 30.0
RELAX_ROUNDS = 12
RELAX_TOLERANCE = 1e-3
# The junction solve's ridge, against unit normals: two lines 5 degrees apart leave the system
# an eigenvalue of 0.004, and a junction between near-parallel lines must not fly away.
RIDGE_EPSILON = 0.02

# Step 7
CREASE_MIN_D = 0.04
CREASE_AVOID_DEG = 10.0
CREASE_LSD_PX = 3.0
CREASE_LSD_DEG = 4.0
CREASE_COVER = 0.60

# Step 8
STANDARD_PITCHES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
PITCH_PRIORS = [0.3, 0.7, 1.6, 1.4, 1.6, 1.0, 1.2, 0.8, 0.8, 1.0, 0.3, 0.2]
PITCH_SIGMA_CALIBRATED_DEG = 2.0
PITCH_SIGMA_APPARENT_DEG = 6.0
PITCH_DISAGREE_DEG = 10.0
PITCH_MAX_DEG = 80.0          # steeper edges are wall steps or cuts, not slopes
# Same angle, same reasoning, applied to the role table instead of the pitch: measured
# on the annotated houses, tightening it below 80 trades F1 for nothing the scorer sees.
WALL_STEP_DEG = 80.0
# A palm trunk leans, and the notch wall the mask takes around it leans with it: 55-78 degrees
# on the client's photo, under the gate above. What still gives it away is that it is short —
# a rake that steep runs to an apex and is long; a notch wall is a stub. Both tests, either fires.
STEEP_STUB_DEG = 55.0
STEEP_STUB_L = 0.10
PLAN_HORIZONTAL_DEG = 6.0
# Dormer rule: a shared sloped border whose smaller plane's free edge lies inside the outline of
# host + child is a dormer's roof edge overhanging the host — a rake with a fascia — not a crease.
DORMER_OUTLINE_PX = 5
DORMER_MIN_FREE = 8
DORMER_INTERIOR_SHARE = 0.80
PLAN_HORIZONTAL_FALLBACK_DEG = 12.0
VALLEY_DISC_L = 0.02
VALLEY_COVER = 0.58
INSIDE_PROBE_L = 0.004
TOP_TO_EAVE = 0.70

# 0 = off, as in the reference, which emits every junction pair of a pixel or more. 0.01 hid
# the slivers T-junctions cut off edges on the 1280px masks; re-measure after detection parity.
MIN_SEGMENT_L = 0.0

# Constrained snap, on the product path: a lit line (eave, rake, hip) may translate along its
# own normal, within the band, to the offset with the strongest gradient support along its whole
# length — soffit to fascia — and no further: direction stays what the graph produced, nothing
# is refit, split or dropped, and every junction is re-solved so the polygons stay closed.
CONSTRAINED_BAND_FRACTION = 0.02   # of the roof extent
CONSTRAINED_STEP = 2.0             # px between samples along the line
CONSTRAINED_BLUR = 1.2
CONSTRAINED_PERCENTILE = 90        # an edge must be as strong as the image's strongest tenth ...
CONSTRAINED_MIN_STRENGTH = 8.0     # ... and this strong outright
CONSTRAINED_NEARER = 1e-3          # tie-break toward the smaller move

# Product: snapping options for this mode, handed to snap.lines by roof_lines. snap.py's 15px
# default is a photo-size constant; here it follows the roof (the largest plane's box).
SNAP_BAND_FRACTION = 0.02
SNAP_SMOOTH_FRACTION = 0.03
SNAP_SIMPLIFY_FRACTION = 0.002


def roof_extent(masks: list[np.ndarray]) -> float:
    """The longer side of the largest plane's box: what the snapping options are a share of."""
    best = 1.0
    for mask in masks:
        rows, columns = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        if rows.size:
            best = max(best, float(rows[-1] - rows[0] + 1), float(columns[-1] - columns[0] + 1))
    return best


def snap_min_points() -> int:
    from app.services import snap
    return snap.MIN_POINTS


def snap_options(extent: float) -> dict:
    return {
        "band": max(3.0, SNAP_BAND_FRACTION * extent),
        "smooth": max(snap_min_points(), int(round(SNAP_SMOOTH_FRACTION * extent / 2.0))),
        "simplify": max(0.5, SNAP_SIMPLIFY_FRACTION * extent),
    }


def _ellipse(size: float) -> np.ndarray:
    k = max(3, int(round(size)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions, sign ignored."""
    return float(np.degrees(np.arccos(min(1.0, abs(float(_unit(a) @ _unit(b)))))))


# -- Step 2. hard partition ---------------------------------------------------------------------


def partition(masks: list[np.ndarray], scores: list[float], L: float) -> np.ndarray:
    """Label map: 0 background, k + 1 plane k. Smallest effective instance wins every pixel;
    an instance left with only rims or crumbs is a duplicate and is removed."""
    height, width = masks[0].shape
    image_area = float(height * width)
    areas = np.array([float(m.sum()) for m in masks])
    effective = []
    for k, area in enumerate(areas):
        score = scores[k] if scores and k < len(scores) and scores[k] else 1.0
        size = area / max(score, 1e-6)
        if score < SCORE_FLOOR:
            size *= SCORE_FLOOR_PENALTY
        effective.append(size)
    order = sorted(range(len(masks)), key=lambda k: (effective[k], k))
    active = [k for k in order if areas[k] > 0]
    rim = _ellipse(RIM_OPEN_L * L)

    while True:
        labels = np.zeros((height, width), dtype=np.int32)
        for k in active:
            labels[masks[k] & (labels == 0)] = k + 1
        duplicate = None
        for k in active:
            kept = labels == k + 1
            if kept.sum() >= KEEP_SHARE * areas[k]:
                continue
            solid = cv2.morphologyEx(kept.astype(np.uint8), cv2.MORPH_OPEN, rim).sum()
            if solid < SOLID_SHARE * areas[k] or solid < PLANE_MIN_IMAGE_SHARE * image_area:
                duplicate = k
                break
        if duplicate is None:
            break
        active.remove(duplicate)

    return _fill_gaps(labels, L)


def _fill_gaps(labels: np.ndarray, L: float) -> np.ndarray:
    """Pixels a closing of the union adds are gaps between neighbours: each goes to the nearest plane."""
    union = (labels > 0).astype(np.uint8)
    if not union.any():
        return labels
    closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, _ellipse(GAP_CLOSE_L * L))
    gaps = (closed > 0) & (union == 0)
    if not gaps.any():
        return labels
    _, nearest = cv2.distanceTransformWithLabels(1 - union, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros(int(nearest.max()) + 1, dtype=np.int32)
    on = union.astype(bool)
    lookup[nearest[on]] = labels[on]
    filled = labels.copy()
    filled[gaps] = lookup[nearest[gaps]]
    return filled


# -- Step 3. cleanup --------------------------------------------------------------------------------


def cleanup(labels: np.ndarray, count: int, L: float) -> np.ndarray:
    """Per plane: close, open, fill holes, drop small blobs, fill narrow notches. Growth only
    ever takes background pixels, so the partition stays hard."""
    height, width = labels.shape
    image_area = float(height * width)
    close_kernel = _ellipse(CLEAN_CLOSE_L * L)
    small = np.ones((3, 3), np.uint8)
    for k in range(count):
        region = (labels == k + 1).astype(np.uint8)
        if not region.any():
            continue
        free = (labels == 0) | (labels == k + 1)
        grown = cv2.morphologyEx(region, cv2.MORPH_CLOSE, close_kernel).astype(bool) & free
        opened = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, small)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(opened)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        opened |= (filled.astype(bool) & free).astype(np.uint8)
        n, components, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        if n > 1:
            largest = stats[1:, cv2.CC_STAT_AREA].max()
            for index in range(1, n):
                if stats[index, cv2.CC_STAT_AREA] < BLOB_MIN_SHARE * largest:
                    opened[components == index] = 0
        if opened.sum() < PLANE_MIN_IMAGE_SHARE * image_area:
            labels[labels == k + 1] = 0
            continue
        opened = _fill_notches(opened, free, L)
        labels[(labels == k + 1) & ~opened.astype(bool)] = 0
        labels[opened.astype(bool) & (labels == 0)] = k + 1
    return labels


def _fill_notches(region: np.ndarray, free: np.ndarray, L: float) -> np.ndarray:
    """Fill a convexity defect when it is deep and narrow: depth >= NOTCH_DEPTH_L, mouth at most
    twice the depth, area at most NOTCH_AREA_SHARE of the plane. Never the whole hull."""
    area = float(region.sum())
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = region.copy()
    for contour in contours:
        if len(contour) < 8:
            continue
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            continue
        try:
            defects = cv2.convexityDefects(contour, hull)
        except cv2.error:
            continue
        if defects is None:
            continue
        for start, end, far, depth in defects[:, 0]:
            depth_px = depth / 256.0
            if depth_px < NOTCH_DEPTH_L * L:
                continue
            a, b = contour[start][0], contour[end][0]
            mouth = float(np.linalg.norm(a - b))
            if mouth > NOTCH_MOUTH_RATIO * depth_px:
                continue
            path = contour[start:end + 1] if end > start else np.vstack([contour[start:], contour[:end + 1]])
            notch = np.zeros_like(region)
            cv2.fillPoly(notch, [path.reshape(-1, 1, 2).astype(np.int32)], 1)
            if float(notch.sum()) > NOTCH_AREA_SHARE * area:
                continue
            out |= (notch.astype(bool) & free).astype(np.uint8)
    return out


def foliage_mask(image: np.ndarray | None, L: float) -> np.ndarray | None:
    """Excess green, cleaned and grown: where a tree or lawn may sit against a roof edge."""
    if image is None:
        return None
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    mask = (exg > EXG_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum = FOLIAGE_MIN_SHARE * mask.size
    for index in range(1, n):
        if stats[index, cv2.CC_STAT_AREA] < minimum:
            mask[components == index] = 0
    return cv2.dilate(mask, _ellipse(FOLIAGE_DILATE_L * L)).astype(bool)


@dataclass
class Boundary:
    plane: int
    contour: np.ndarray          # (N, 2) int, every boundary pixel, in order
    tags: np.ndarray             # (N,) int: 0 free, else neighbour plane index + 1
    trusted: np.ndarray          # (N,) bool


def boundaries(labels: np.ndarray, count: int, L: float, suspect: np.ndarray | None,
               blocked: np.ndarray | None = None) -> list[Boundary]:
    """Every plane's boundary pixels, tagged with the plane across them and marked trusted.

    `suspect` is the colour guess at foliage: a boundary there is doubted, then kept where it is
    straight and long enough to be a roof edge. `blocked` is a detected object standing in front
    of the roof: a boundary there is never trusted, straight or not. That distinction is the
    whole point — the edge of a palm trunk is perfectly straight, which is exactly why the
    straightness test used to keep it and draw it as a hip.
    """
    height, width = labels.shape
    out: list[Boundary] = []
    tag_kernel = np.ones((2 * TAG_REACH + 1, 2 * TAG_REACH + 1), np.uint8)
    for k in range(count):
        region = (labels == k + 1).astype(np.uint8)
        if not region.any():
            continue
        others = np.where((labels > 0) & (labels != k + 1), labels, 0).astype(np.float32)
        near = cv2.dilate(others, tag_kernel)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            trace = contour.reshape(-1, 2)
            if len(trace) < 4 * TAG_MIN_RUN:
                continue
            tags = _smooth_runs(near[trace[:, 1], trace[:, 0]].astype(np.int32), TAG_MIN_RUN)
            frame = ((trace[:, 0] < FRAME_PX) | (trace[:, 0] >= width - FRAME_PX)
                     | (trace[:, 1] < FRAME_PX) | (trace[:, 1] >= height - FRAME_PX))
            trusted = ~frame
            if suspect is not None:
                flagged = suspect[trace[:, 1], trace[:, 0]] & ~frame
                trusted &= _trust_suspects(trace, flagged, len(trace))
            if blocked is not None:
                trusted &= ~blocked[trace[:, 1], trace[:, 0]]
            out.append(Boundary(plane=k, contour=trace, tags=tags, trusted=trusted))
    return out


def _smooth_runs(values: np.ndarray, minimum: int) -> np.ndarray:
    values = values.copy()
    for _ in range(3):
        changes = np.flatnonzero(values != np.roll(values, 1))
        if len(changes) < 2:
            return values
        changed = False
        for start, following in zip(changes, np.roll(changes, -1)):
            length = (following - start) % len(values)
            if 0 < length < minimum:
                values[(start + np.arange(length)) % len(values)] = values[start - 1]
                changed = True
        if not changed:
            break
    return values


def _trust_suspects(trace: np.ndarray, flagged: np.ndarray, perimeter: int) -> np.ndarray:
    """Foliage beside a boundary makes it suspect, not untrusted. Each suspect stretch is split
    at the point farthest from its chord until every piece is straight or too short; straight
    pieces long enough to be a roof edge keep their trust, the rest is a tree's outline."""
    trusted = np.ones(len(trace), dtype=bool)
    if not flagged.any():
        return trusted
    minimum = max(TRUSTED_MIN_PX, TRUSTED_MIN_PERIMETER * perimeter)
    changes = np.flatnonzero(flagged != np.roll(flagged, 1))
    if len(changes) == 0:
        stretches = [np.arange(len(trace))]
    else:
        stretches = []
        for start, following in zip(changes, np.roll(changes, -1)):
            if not flagged[start]:
                continue
            length = (following - start) % len(trace) or len(trace)
            stretches.append((start + np.arange(length)) % len(trace))
    for indices in stretches:
        for piece, straight in _straight_pieces(trace[indices].astype(float)):
            if not (straight and len(piece) >= minimum):
                trusted[indices[piece]] = False
    return trusted


def _straight_pieces(points: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """(index array, is straight) pieces of a stretch, split recursively at the farthest point."""
    def recurse(index: np.ndarray):
        pts = points[index]
        if len(pts) < 3:
            return [(index, False)]
        chord = pts[-1] - pts[0]
        length = float(np.linalg.norm(chord))
        if length < 1e-6:
            return [(index, False)]
        normal = np.array([-chord[1], chord[0]]) / length
        deviation = np.abs((pts - pts[0]) @ normal)
        centre = pts.mean(axis=0)
        axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
        rms = float(np.sqrt(np.mean(((pts - centre) @ np.array([-axis[1], axis[0]])) ** 2)))
        if rms <= STRAIGHT_RMS:
            return [(index, True)]
        split = int(np.argmax(deviation))
        if split <= 1 or split >= len(pts) - 2:
            return [(index, False)]
        return recurse(index[:split + 1]) + recurse(index[split:])
    return recurse(np.arange(len(points)))


# -- Step 4. lines per plane ---------------------------------------------------------------------


@dataclass
class Line:
    moments: np.ndarray          # (6,): w, wx, wy, wxx, wxy, wyy
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    first: np.ndarray = field(default_factory=lambda: np.zeros(2))   # first supporting pixel along travel
    last: np.ndarray = field(default_factory=lambda: np.zeros(2))
    tags: dict = field(default_factory=dict)
    prior: np.ndarray | None = None
    prior_weight: float = PRIOR_LAMBDA
    plane: int = -1
    group: int = -1              # shared line id after step 6

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.last - self.first))

    @property
    def neighbour(self) -> int:
        """Plane index across this edge, or -1 when free (by pixel majority)."""
        if not self.tags:
            return -1
        tag = max(self.tags, key=self.tags.get)
        return tag - 1 if tag > 0 and self.tags[tag] > sum(self.tags.values()) / 2 else -1


def _moments(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(points)) if weights is None else weights
    x, y = points[:, 0], points[:, 1]
    return np.array([w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum(), (w * y * y).sum()])


def _fit_moments(moments: np.ndarray, prior: np.ndarray | None, span: float,
                 weight: float = PRIOR_LAMBDA) -> tuple[np.ndarray, np.ndarray]:
    """Total least squares from moments, the direction prior as scatter + lambda u0 u0^T on
    the normal: centroid and unit direction."""
    w, sx, sy, sxx, sxy, syy = moments
    if w <= 0:
        return np.zeros(2), np.array([1.0, 0.0])
    cx, cy = sx / w, sy / w
    scatter = np.array([[sxx - w * cx * cx, sxy - w * cx * cy], [sxy - w * cx * cy, syy - w * cy * cy]])
    if prior is not None:
        stiffness = w * span * span / 12.0
        scatter = scatter + weight * stiffness * np.outer(prior, prior)
    values, vectors = np.linalg.eigh(scatter)
    normal = vectors[:, int(np.argmin(values))]
    return np.array([cx, cy]), _unit(np.array([normal[1], -normal[0]]))


def _refit(line: Line, extra_points: np.ndarray | None = None, extra_weight: float = 0.0) -> None:
    moments = line.moments.copy()
    if extra_points is not None and len(extra_points):
        moments = moments + _moments(extra_points, np.full(len(extra_points), extra_weight))
    travel = line.last - line.first
    origin, direction = _fit_moments(moments, line.prior, max(line.length, 1.0), line.prior_weight)
    if float(direction @ travel) < 0:
        direction = -direction
    line.origin, line.direction = origin, direction


def _project(line: Line, point: np.ndarray) -> np.ndarray:
    return line.origin + line.direction * float((point - line.origin) @ line.direction)


def _distance(line: Line, point: np.ndarray) -> float:
    return abs(float((point - line.origin) @ line.normal))


def _intersect(a: Line, b: Line) -> np.ndarray | None:
    matrix = np.stack([a.normal, b.normal])
    if abs(np.linalg.det(matrix)) < 1e-3:
        return None
    return np.linalg.solve(matrix, np.array([a.normal @ a.origin, b.normal @ b.origin]))


def _seed_indices(boundary: Boundary, eps: float) -> list[int]:
    """Douglas-Peucker corners plus a seed wherever the tag changes, unless one is within eps/2."""
    trace = boundary.contour
    corners = cv2.approxPolyDP(trace.reshape(-1, 1, 2).astype(np.int32), eps, True).reshape(-1, 2)
    seeds: list[int] = []
    for corner in corners:
        matches = np.flatnonzero((trace == corner).all(axis=1))
        if len(matches):
            seeds.append(int(matches[0]))
    for index in np.flatnonzero(boundary.tags != np.roll(boundary.tags, 1)):
        point = trace[index].astype(float)
        if all(np.linalg.norm(point - trace[s]) > eps / 2 for s in seeds):
            seeds.append(int(index))
    return sorted(set(seeds))


@dataclass
class Run:
    start: int                   # seed index at the start (inclusive)
    end: int                     # seed index at the end (inclusive)
    line: Line | None            # None for a gap


def _fit_runs(boundary: Boundary, seeds: list[int]) -> list[Run]:
    trace, trusted, tags = boundary.contour.astype(float), boundary.trusted, boundary.tags
    runs: list[Run] = []
    for i, start in enumerate(seeds):
        end = seeds[(i + 1) % len(seeds)]
        if end > start:
            index = np.arange(start, end + 1)
        else:
            index = np.concatenate([np.arange(start, len(trace)), np.arange(0, end + 1)])
        pts, ok = trace[index], trusted[index]
        if ok.sum() < max(RUN_MIN_TRUSTED, RUN_MIN_TRUSTED_SHARE * len(index)):
            runs.append(Run(start, end, None))
            continue
        used = pts[ok]
        vx, vy, x0, y0 = cv2.fitLine(used.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        normal = np.array([-vy, vx], dtype=float)
        residual = np.abs((used - np.array([x0, y0])) @ normal)
        kept = used[residual <= FIT_TRIM_PX]
        if len(kept) < 2:
            kept = used
        line = Line(moments=_moments(kept), plane=boundary.plane)
        line.first, line.last = kept[0], kept[-1]
        _refit(line)
        along = (kept - line.origin) @ line.direction
        line.first, line.last = kept[int(np.argmin(along))], kept[int(np.argmax(along))]
        if float((line.last - line.first) @ (pts[-1] - pts[0])) < 0:
            line.first, line.last = line.last, line.first
            line.direction = -line.direction
        values, counts = np.unique(tags[index][ok], return_counts=True)
        line.tags = {int(v): int(c) for v, c in zip(values, counts)}
        runs.append(Run(start, end, line))
    return runs


def _merge_runs(runs: list[Run], eps: float) -> list[Run]:
    """Merge consecutive lines within MERGE_DEG and collinear, smallest angle first, even across
    a gap; stop at MERGE_MIN_EDGES."""
    runs = list(runs)
    while sum(1 for r in runs if r.line is not None) > MERGE_MIN_EDGES:
        lined = [i for i, r in enumerate(runs) if r.line is not None]
        best = None
        for position, i in enumerate(lined):
            j = lined[(position + 1) % len(lined)]
            if i == j:
                break
            a, b = runs[i].line, runs[j].line
            angle = _angle_between(a.direction, b.direction)
            tolerance = max(3.0, eps)
            if angle < MERGE_DEG and _distance(a, b.origin) <= tolerance and _distance(b, a.origin) <= tolerance:
                if best is None or angle < best[0]:
                    best = (angle, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = runs[i].line, runs[j].line
        merged = Line(moments=a.moments + b.moments, plane=a.plane, prior=a.prior if a.prior is not None else b.prior)
        merged.first, merged.last = a.first, b.last
        merged.tags = dict(a.tags)
        for tag, n in b.tags.items():
            merged.tags[tag] = merged.tags.get(tag, 0) + n
        _refit(merged)
        if j > i:
            runs = runs[:i] + [Run(runs[i].start, runs[j].end, merged)] + runs[j + 1:]
        else:
            runs = runs[j + 1:i] + [Run(runs[i].start, runs[j].end, merged)]
    return runs


@dataclass
class Edge:
    line: Line | None            # None for an inferred connector
    start: np.ndarray
    end: np.ndarray
    plane: int
    neighbour: int               # plane across, or -1
    inferred: bool = False


def _close_polygon(boundary: Boundary, runs: list[Run], eps: float, L: float) -> list[Edge]:
    """Corners are line intersections. Across a gap the intersection is accepted when it lies
    ahead of the first line, before the second and within reach of both; otherwise the gap is
    bridged by a connector flagged inferred."""
    positions = [i for i, r in enumerate(runs) if r.line is not None]
    if len(positions) < 2:
        return []
    trace = boundary.contour.astype(float)
    reach = max(2 * eps, CORNER_REACH_L * L)
    count = len(positions)
    starts: list = [None] * count
    ends: list = [None] * count
    connectors = [False] * count
    for n, (i, j) in enumerate(zip(positions, positions[1:] + positions[:1])):
        a, b = runs[i].line, runs[j].line
        gap = (j - i) % len(runs) != 1
        cross = _intersect(a, b)
        following = (n + 1) % count
        if not gap:
            seed = trace[runs[j].start]
            if cross is not None and np.linalg.norm(cross - seed) <= reach:
                corner = cross
            else:
                corner = (_project(a, seed) + _project(b, seed)) / 2
            ends[n], starts[following] = corner, corner
            continue
        length = float(np.linalg.norm(b.first - a.last))
        accept = (cross is not None and float((cross - a.last) @ a.direction) > 0
                  and float((b.first - cross) @ b.direction) > 0
                  and np.linalg.norm(cross - a.last) <= GAP_REACH * length
                  and np.linalg.norm(cross - b.first) <= GAP_REACH * length)
        if accept:
            ends[n], starts[following] = cross, cross
        else:
            ends[n] = _project(a, a.last)
            starts[following] = _project(b, b.first)
            connectors[n] = True
    edges: list[Edge] = []
    for n, i in enumerate(positions):
        line = runs[i].line
        edges.append(Edge(line=line, start=starts[n], end=ends[n], plane=boundary.plane, neighbour=line.neighbour))
        if connectors[n]:
            edges.append(Edge(line=None, start=ends[n], end=starts[(n + 1) % count],
                              plane=boundary.plane, neighbour=-1, inferred=True))
    return edges


def plane_polygons(bounds: list[Boundary], L: float) -> list[list[Edge]]:
    """Step 4 for every plane boundary."""
    perimeters = [len(b.contour) for b in bounds]
    largest = max(perimeters) if perimeters else 1
    polygons: list[list[Edge]] = []
    for boundary in bounds:
        eps = max(1.0, min(SEED_EPS_LARGEST * largest, SEED_EPS_OWN * len(boundary.contour)))
        seeds = _seed_indices(boundary, eps)
        if len(seeds) < 2:
            continue
        runs = _merge_runs(_fit_runs(boundary, seeds), eps)
        polygon = _close_polygon(boundary, runs, eps, L)
        if polygon:
            polygons.append(polygon)
    return polygons


# -- Step 5. vanishing points ---------------------------------------------------------------------


@dataclass
class Scene:
    segments: np.ndarray          # (M, 4) x1 y1 x2 y2 in working coordinates
    vertical: np.ndarray | None   # homogeneous VP in normalised coordinates, or None
    horizontal: list              # trusted horizontal VPs, homogeneous normalised
    centre: np.ndarray
    scale: float


def lsd_segments(gray: np.ndarray, foliage: np.ndarray | None) -> np.ndarray:
    """Line segments of the whole photo at LSD_LONG_EDGE, returned in working coordinates."""
    height, width = gray.shape
    factor = min(1.0, LSD_LONG_EDGE / max(height, width))
    small = cv2.resize(gray, (max(1, round(width * factor)), max(1, round(height * factor)))) if factor < 1 else gray
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(small)[0]
    if found is None or not len(found):
        return np.zeros((0, 4))
    segments = found.reshape(-1, 4).astype(float) / factor
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    segments = segments[lengths >= LSD_MIN_D * np.hypot(width, height)]
    if foliage is not None and len(segments):
        xs = np.clip(np.rint((segments[:, 0] + segments[:, 2]) / 2).astype(int), 0, width - 1)
        ys = np.clip(np.rint((segments[:, 1] + segments[:, 3]) / 2).astype(int), 0, height - 1)
        segments = segments[~foliage[ys, xs]]
    return segments


def _vp_direction(vp: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Direction toward a homogeneous VP from a normalised point; a VP at infinity is a direction."""
    if abs(vp[2]) < 1e-9:
        return _unit(vp[:2])
    return _unit(vp[:2] / vp[2] - point)


def _agreement(mids: np.ndarray, dirs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Per segment: does the direction from its midpoint to the VP match its own direction?"""
    if abs(vp[2]) < 1e-9:
        toward = np.tile(_unit(vp[:2]), (len(mids), 1))
    else:
        toward = vp[:2] / vp[2] - mids
        toward = toward / np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
    return np.abs(np.einsum("ij,ij->i", toward, dirs)) >= np.cos(np.radians(VP_AGREE_DEG))


def _refine_vp(lines: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.linalg.svd(lines * np.sqrt(weights)[:, None], full_matrices=False)[2][-1]


def vanishing_points(segments: np.ndarray, shape: tuple[int, int]) -> Scene:
    height, width = shape
    centre = np.array([width / 2.0, height / 2.0])
    scale = max(width, height) / 2.0
    scene = Scene(segments=segments, vertical=None, horizontal=[], centre=centre, scale=scale)
    if len(segments) < 8:
        return scene
    p1 = np.hstack([(segments[:, :2] - centre) / scale, np.ones((len(segments), 1))])
    p2 = np.hstack([(segments[:, 2:] - centre) / scale, np.ones((len(segments), 1))])
    lines = np.cross(p1, p2)
    lines = lines / np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-9)
    mids = (p1[:, :2] + p2[:, :2]) / 2
    dirs = p2[:, :2] - p1[:, :2]
    lengths = np.linalg.norm(dirs, axis=1)
    dirs = dirs / np.maximum(lengths[:, None], 1e-9)
    total = float(lengths.sum())
    rng = np.random.default_rng(0)

    def best_of(candidates: np.ndarray):
        best, best_score = None, 0.0
        for vp in candidates:
            score = float(lengths[_agreement(mids, dirs, vp)].sum())
            if score > best_score:
                best, best_score = vp, score
        return best, best_score

    upright = np.flatnonzero(np.abs(dirs[:, 1]) >= np.cos(np.radians(VP_UPRIGHT_DEG)))
    if len(upright) >= 2:
        pairs = rng.choice(upright, size=(VP_RANSAC_ROUNDS, 2))
        candidates = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
        candidates = candidates[np.linalg.norm(candidates, axis=1) > 1e-9]
        vertical, score = best_of(candidates)
        if vertical is not None:
            inliers = _agreement(mids, dirs, vertical)
            if inliers.sum() >= 2:
                vertical = _refine_vp(lines[inliers], lengths[inliers])
                score = float(lengths[_agreement(mids, dirs, vertical)].sum())
            if score >= VP_TRUST_SHARE * total:
                scene.vertical = vertical

    level = np.flatnonzero(np.abs(dirs[:, 1]) < np.cos(np.radians(VP_UPRIGHT_DEG)))
    if len(level) < 2:
        return scene
    pairs = rng.choice(level, size=(VP_CANDIDATES, 2))
    candidates = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
    candidates = candidates[np.abs(candidates[:, 2]) > 1e-9]
    candidates = candidates[:, :2] / candidates[:, 2:3]
    best_pair, best_score = None, 0.0
    if scene.vertical is not None and abs(scene.vertical[2]) > 1e-9:
        v = scene.vertical[:2] / scene.vertical[2]
        normal = _unit(v)
        along = np.array([-normal[1], normal[0]])
        for f in np.linspace(0.5, 3.0, VP_FOCALS) * (width / scale):
            # Horizon for K = diag(f, f, 1): the polar of the vertical VP, v . x = -f^2.
            offset = -f * f / max(float(np.linalg.norm(v)), 1e-9)
            foot = normal * offset
            for c in candidates:
                v1 = foot + along * float((c - foot) @ along)
                denominator = float(v1 @ along)
                if abs(denominator) < 1e-9:
                    continue
                v2 = v1 + along * ((-f * f - float(v1 @ v1)) / denominator)
                vp1, vp2 = np.array([v1[0], v1[1], 1.0]), np.array([v2[0], v2[1], 1.0])
                score = float(lengths[_agreement(mids, dirs, vp1) | _agreement(mids, dirs, vp2)].sum())
                if score > best_score:
                    best_pair, best_score = (vp1, vp2), score
    else:
        vp, score = best_of(np.hstack([candidates, np.ones((len(candidates), 1))]))
        if vp is not None:
            best_pair, best_score = (vp,), score
    if best_pair is None:
        return scene
    for vp in best_pair:
        inliers = _agreement(mids, dirs, vp)
        if inliers.sum() >= 2:
            vp = _refine_vp(lines[inliers], lengths[inliers])
        if float(lengths[_agreement(mids, dirs, vp)].sum()) >= VP_TRUST_SHARE * total:
            scene.horizontal.append(vp)
    return scene


def _vp_directions(scene: Scene, point: np.ndarray, horizontal_only: bool = False) -> list[np.ndarray]:
    """Image-space directions toward the trusted VPs from a working-coordinate point."""
    normalised = (point - scene.centre) / scene.scale
    vps = list(scene.horizontal)
    if not horizontal_only and scene.vertical is not None:
        vps.append(scene.vertical)
    return [_vp_direction(vp, normalised) for vp in vps]


def _prior_at(scene: Scene, direction: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    best, best_angle = None, PRIOR_DEG
    for candidate in _vp_directions(scene, point):
        angle = _angle_between(direction, candidate)
        if angle <= best_angle:
            best, best_angle = candidate, angle
    return best


def apply_priors(polygons: list[list[Edge]], scene: Scene) -> None:
    """A roof edge within PRIOR_DEG of a VP direction is refit toward it."""
    for polygon in polygons:
        for edge in polygon:
            if edge.line is None:
                continue
            edge.line.prior = _prior_at(scene, edge.line.direction, (edge.start + edge.end) / 2)
            if edge.line.prior is not None:
                _refit(edge.line)


def apply_cluster_priors(polygons: list[list[Edge]]) -> None:
    """No usable vanishing point: near-parallel edges share a direction. Greedy clusters on
    doubled angles, stiffness-weighted means, applied at CLUSTER_LAMBDA."""
    lines = [e.line for polygon in polygons for e in polygon if e.line is not None]
    order = sorted(lines, key=lambda l: -(l.moments[0] * max(l.length, 1.0) ** 2))
    clusters: list[dict] = []
    for line in order:
        stiffness = float(line.moments[0] * max(line.length, 1.0) ** 2 / 12.0)
        doubled = 2.0 * np.arctan2(line.direction[1], line.direction[0])
        vector = stiffness * np.array([np.cos(doubled), np.sin(doubled)])
        for cluster in clusters:
            mean = np.arctan2(cluster["sum"][1], cluster["sum"][0])
            if abs((doubled - mean + np.pi) % (2 * np.pi) - np.pi) <= np.radians(2 * CLUSTER_DEG):
                cluster["sum"] = cluster["sum"] + vector
                cluster["members"].append(line)
                break
        else:
            clusters.append({"sum": vector, "members": [line]})
    for cluster in clusters:
        if len(cluster["members"]) < 2:
            continue
        mean = np.arctan2(cluster["sum"][1], cluster["sum"][0]) / 2.0
        prior = np.array([np.cos(mean), np.sin(mean)])
        for line in cluster["members"]:
            line.prior, line.prior_weight = prior, CLUSTER_LAMBDA
            _refit(line)


# -- Step 6. one planar graph -----------------------------------------------------------------------


@dataclass
class Junction:
    point: np.ndarray
    lines: set = field(default_factory=set)


def build_graph(polygons: list[list[Edge]], scene: Scene, L: float):
    """Shared lines merged across planes, corners clustered into junctions, then the energy
    E = sum n^T S n + priors + JUNCTION_WEIGHT * sum dist(junction, line)^2 relaxed by block
    coordinate descent."""
    edges = [e for polygon in polygons for e in polygon if e.line is not None]
    lines = [e.line for e in edges]
    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a, b = lines[i], lines[j]
            if _angle_between(a.direction, b.direction) >= LINE_MERGE_DEG:
                continue
            if _distance(a, b.origin) > LINE_MERGE_OFFSET_L * L or _distance(b, a.origin) > LINE_MERGE_OFFSET_L * L:
                continue
            ai = sorted([float((a.first - a.origin) @ a.direction), float((a.last - a.origin) @ a.direction)])
            bi = sorted([float((b.first - a.origin) @ a.direction), float((b.last - a.origin) @ a.direction)])
            if max(ai[0], bi[0]) - min(ai[1], bi[1]) > LINE_MERGE_GAP_L * L:
                continue
            parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    shared: list[Line] = []
    for members in groups.values():
        first = lines[members[0]]
        merged = Line(moments=sum(lines[m].moments for m in members), plane=first.plane)
        merged.first, merged.last = first.first, first.last
        priors = [lines[m].prior for m in members if lines[m].prior is not None]
        merged.prior = priors[0] if priors else None
        _refit(merged)
        ends = np.array([p for m in members for p in (lines[m].first, lines[m].last)])
        along = (ends - merged.origin) @ merged.direction
        merged.first, merged.last = ends[int(np.argmin(along))], ends[int(np.argmax(along))]
        merged.group = len(shared)
        for m in members:
            lines[m].group = merged.group
        shared.append(merged)

    radius = JUNCTION_L * L
    corners = [(e.start, e) for e in edges] + [(e.end, e) for e in edges]
    points = np.array([c[0] for c in corners])
    assigned = np.full(len(points), -1)
    clusters: list[list[int]] = []
    def meets(i: int, j: int) -> bool:
        """Two corners can be one junction when their lines are the same line or cross at a
        real angle. The eave and ridge corners of a thin band sit within the radius of each
        other on near-parallel lines; pulling them together would rotate both lines."""
        if JUNCTION_MIN_ANGLE_DEG <= 0:
            return True
        a, b = corners[i][1].line, corners[j][1].line
        return a.group == b.group or _angle_between(a.direction, b.direction) >= JUNCTION_MIN_ANGLE_DEG

    for i in range(len(points)):
        if assigned[i] >= 0:
            continue
        cluster = [i]
        assigned[i] = len(clusters)
        for j in range(i + 1, len(points)):
            if assigned[j] >= 0 or np.linalg.norm(points[j] - points[i]) > radius:
                continue
            if all(np.linalg.norm(points[j] - points[m]) <= 2 * radius for m in cluster) \
                    and any(meets(m, j) for m in cluster):
                cluster.append(j)
                assigned[j] = len(clusters)
        clusters.append(cluster)
    junctions: list[Junction] = []
    for cluster in clusters:
        junction = Junction(point=points[cluster].mean(axis=0))
        for m in cluster:
            junction.lines.add(corners[m][1].line.group)
        junctions.append(junction)
    for junction in junctions:
        for e in edges:
            g = e.line.group
            if g in junction.lines:
                continue
            group = shared[g]
            span = sorted([float((e.start - group.origin) @ group.direction), float((e.end - group.origin) @ group.direction)])
            along = float((junction.point - group.origin) @ group.direction)
            if _distance(group, junction.point) <= radius and span[0] + radius < along < span[1] - radius:
                junction.lines.add(g)

    def energy() -> float:
        total = 0.0
        for line in shared:
            w, sx, sy, sxx, sxy, syy = line.moments
            n = line.normal
            cx, cy = (sx / w, sy / w) if w else (0.0, 0.0)
            scatter = np.array([[sxx - w * cx * cx, sxy - w * cx * cy], [sxy - w * cx * cy, syy - w * cy * cy]])
            total += float(n @ scatter @ n)
            if line.prior is not None:
                total += line.prior_weight * w * max(line.length, 1.0) ** 2 / 12.0 * float(line.prior @ n) ** 2
        for junction in junctions:
            for g in junction.lines:
                total += JUNCTION_WEIGHT * _distance(shared[g], junction.point) ** 2
        return total

    previous = energy()
    for _ in range(RELAX_ROUNDS):
        for g, line in enumerate(shared):
            attached = np.array([j.point for j in junctions if g in j.lines])
            extra = attached if len(attached) else None
            _refit(line, extra, JUNCTION_WEIGHT)
            if scene.horizontal or scene.vertical is not None:
                prior = _prior_at(scene, line.direction, line.origin)
                if prior is not None:
                    line.prior = prior
                    _refit(line, extra, JUNCTION_WEIGHT)
        for junction in junctions:
            if len(junction.lines) < 2:
                if junction.lines:
                    junction.point = _project(shared[next(iter(junction.lines))], junction.point)
                continue
            matrix = RIDGE_EPSILON * np.eye(2)
            rhs = RIDGE_EPSILON * junction.point
            for g in junction.lines:
                n = shared[g].normal
                matrix += np.outer(n, n)
                rhs += n * float(n @ shared[g].origin)
            junction.point = np.linalg.solve(matrix, rhs)
        current = energy()
        if previous > 0 and (previous - current) / previous < RELAX_TOLERANCE:
            break
        previous = current
    return edges, shared, junctions, assigned


@dataclass
class Segment:
    start: np.ndarray
    end: np.ndarray
    planes: tuple               # one plane, or the two it lies between
    role: str = ""
    inferred: bool = False      # a connector across an occluded stretch, drawn from no pixels
    key: tuple | None = None    # (line group, junction, junction) for segments from the graph


def graph_segments(edges, shared, junctions, assigned) -> list[Segment]:
    """Walk every polygon edge through its junctions; each junction pair once; a border shared
    by two planes is one segment."""
    count = len(edges)
    emitted: dict[tuple[int, int, int], Segment] = {}
    for i, e in enumerate(edges):
        g = e.line.group
        line = shared[g]
        a, b = assigned[i], assigned[i + count]
        along = [float((junctions[j].point - line.origin) @ line.direction) for j in (a, b)]
        low, high = min(along), max(along)
        between = sorted(
            (float((j.point - line.origin) @ line.direction), idx) for idx, j in enumerate(junctions)
            if g in j.lines and low - 1e-6 <= float((j.point - line.origin) @ line.direction) <= high + 1e-6
        )
        planes = {e.plane} | ({e.neighbour} if e.neighbour >= 0 else set())
        for (_, ja), (_, jb) in zip(between, between[1:]):
            key = (g, min(ja, jb), max(ja, jb))
            if key in emitted:
                emitted[key].planes = tuple(sorted(set(emitted[key].planes) | planes))
                continue
            emitted[key] = Segment(start=_project(line, junctions[ja].point), end=_project(line, junctions[jb].point),
                                   planes=tuple(sorted(planes)), key=key)
    return list(emitted.values())


# -- Step 7. creases a single mask hid ---------------------------------------------------------------


def interior_creases(polygons: list[list[Edge]], labels: np.ndarray, scene: Scene, L: float, D: float) -> list[Segment]:
    """A chord between two non-adjacent corners, inside the plane, long enough, away from the
    plane's near-horizontal edges, and covered at least CREASE_COVER by LSD segments."""
    found: list[Segment] = []
    if not len(scene.segments):
        return found
    seg = scene.segments
    seg_dir = seg[:, 2:] - seg[:, :2]
    seg_dir = seg_dir / np.maximum(np.linalg.norm(seg_dir, axis=1, keepdims=True), 1e-9)
    height, width = labels.shape
    for polygon in polygons:
        real = [e for e in polygon if e.line is not None]
        if len(real) < 4:
            continue
        plane = real[0].plane
        corners = [e.start for e in real]
        horizontals = [e.line.direction for e in real
                       if _is_plan_horizontal(e.line.direction, (e.start + e.end) / 2, scene)]
        best = None
        for i in range(len(corners)):
            for j in range(i + 2, len(corners)):
                if i == 0 and j == len(corners) - 1:
                    continue
                a, b = corners[i], corners[j]
                chord = b - a
                length = float(np.linalg.norm(chord))
                if length < CREASE_MIN_D * D:
                    continue
                direction = chord / length
                if any(_angle_between(direction, h) < CREASE_AVOID_DEG for h in horizontals):
                    continue
                inside = True
                for t in (0.25, 0.5, 0.75):
                    p = a + chord * t
                    x, y = int(round(p[0])), int(round(p[1]))
                    if not (0 <= x < width and 0 <= y < height) or labels[y, x] != plane + 1:
                        inside = False
                        break
                if not inside:
                    continue
                normal = np.array([-direction[1], direction[0]])
                close = ((np.abs((seg[:, :2] - a) @ normal) <= CREASE_LSD_PX)
                         & (np.abs((seg[:, 2:] - a) @ normal) <= CREASE_LSD_PX))
                aligned = np.abs(seg_dir @ direction) >= np.cos(np.radians(CREASE_LSD_DEG))
                use = close & aligned
                if not use.any():
                    continue
                starts_ = (seg[use, :2] - a) @ direction
                ends_ = (seg[use, 2:] - a) @ direction
                intervals = sorted((min(s, e), max(s, e)) for s, e in zip(starts_, ends_))
                covered, cursor = 0.0, 0.0
                for s, e in intervals:
                    s, e = max(s, cursor, 0.0), min(e, length)
                    if e > s:
                        covered += e - s
                        cursor = e
                cover = covered / length
                if cover >= CREASE_COVER and (best is None or cover > best[0]):
                    best = (cover, a, b)
        if best is not None:
            found.append(Segment(start=best[1], end=best[2], planes=(plane, plane)))
    return found


# -- Step 8. roles -----------------------------------------------------------------------------------


def _is_plan_horizontal(direction: np.ndarray, point: np.ndarray, scene: Scene) -> bool:
    horizontals = _vp_directions(scene, point, horizontal_only=True)
    if horizontals:
        return min(_angle_between(direction, h) for h in horizontals) <= PLAN_HORIZONTAL_DEG
    return _angle_between(direction, np.array([1.0, 0.0])) <= PLAN_HORIZONTAL_FALLBACK_DEG


def _is_wall_step(direction: np.ndarray, point: np.ndarray, scene: Scene, length: float, L: float) -> bool:
    """Is this edge too steep to be a roof edge at all?

    Same test plane_pitches uses to refuse a pitch: past PITCH_MAX_DEG from the plan horizontal
    an edge is a wall step or a cut, not a slope. On the roof outline that means the side of a
    step between two storeys, or the wall of a notch the mask took around whatever stands in
    front of the roof. Left lit, those notch walls are what reads as "the boundary line is the
    roofline", and they break the eave at every trunk. A leaning trunk leans its notch walls
    under that gate, so a steep *stub* is refused as well: see STEEP_STUB_DEG.
    """
    horizontals = _vp_directions(scene, point, horizontal_only=True)
    reference = (min(horizontals, key=lambda h: _angle_between(direction, h))
                 if horizontals else np.array([1.0, 0.0]))
    return _steep_stub(_angle_between(direction, reference), length, L)


def _steep_stub(angle: float, length: float, L: float) -> bool:
    return angle > WALL_STEP_DEG or (angle > STEEP_STUB_DEG and length < STEEP_STUB_L * L)


def _plane_above(segment: Segment, labels: np.ndarray, plane: int, L: float) -> bool:
    """Does the plane lie above this free edge? Tested a little inside along the normal."""
    height, width = labels.shape
    mid = (segment.start + segment.end) / 2
    direction = _unit(segment.end - segment.start)
    normal = np.array([-direction[1], direction[0]])
    probe = max(2.0, INSIDE_PROBE_L * L)
    for sign in (1.0, -1.0):
        p = mid + normal * probe * sign
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < width and 0 <= y < height and labels[y, x] == plane + 1:
            return p[1] < mid[1]
    return True


def _valley(segment: Segment, labels: np.ndarray, L: float) -> bool:
    height, width = labels.shape
    foot = segment.start if segment.start[1] > segment.end[1] else segment.end
    radius = max(3, int(VALLEY_DISC_L * L))
    x0, x1 = max(0, int(foot[0]) - radius), min(width, int(foot[0]) + radius + 1)
    y0, y1 = max(0, int(foot[1]) - radius), min(height, int(foot[1]) + radius + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    disc = (xs - foot[0]) ** 2 + (ys - foot[1]) ** 2 <= radius ** 2
    if not disc.any():
        return False
    return float((labels[y0:y1, x0:x1][disc] > 0).mean()) > VALLEY_COVER


class Dormers:
    """Which (child, host) plane pairs are a dormer on a roof: the child's free boundary pixels
    lie inside the outline of host + child rather than on it. A wing's or a nested hip roof's
    free edges sit on the outer outline, so they keep the crease rules."""

    def __init__(self, labels: np.ndarray, bounds: list[Boundary]):
        self.labels = labels
        self.free = {}
        for b in bounds:
            pts = b.contour[b.tags == 0]
            if len(pts):
                self.free.setdefault(b.plane, []).append(pts)
        self.areas = {k: int((labels == k + 1).sum()) for k in range(int(labels.max()))}
        self.cache: dict[tuple[int, int], bool] = {}

    def child_host(self, a: int, b: int) -> tuple[int, int]:
        return (a, b) if self.areas.get(a, 0) <= self.areas.get(b, 0) else (b, a)

    def enclosed(self, child: int, host: int) -> bool:
        key = (child, host)
        if key in self.cache:
            return self.cache[key]
        free = np.vstack(self.free[child]) if child in self.free else np.zeros((0, 2), int)
        result = False
        if len(free) >= DORMER_MIN_FREE:
            union = ((self.labels == child + 1) | (self.labels == host + 1)).astype(np.uint8)
            contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = np.zeros_like(union)
            cv2.drawContours(outline, contours, -1, 1, DORMER_OUTLINE_PX)
            on_outline = outline[free[:, 1], free[:, 0]].astype(bool)
            result = float((~on_outline).mean()) >= DORMER_INTERIOR_SHARE
        self.cache[key] = result
        return result


def assign_roles(segments: list[Segment], labels: np.ndarray, scene: Scene, L: float,
                 dormers: "Dormers | None" = None) -> None:
    """The role table: anything steeper than a roof can be -> junction (a wall step or a notch
    wall, never lit); then shared horizontal -> ridge; shared sloped -> rake when the smaller plane
    is a dormer on the other, else valley at a re-entrant foot, else hip; free horizontal -> eave
    with the plane above, ridge below; free sloped -> hip on a plane whose top is under
    TOP_TO_EAVE of its eave (an apex counts as 0), else rake."""
    eave_length: dict[int, float] = {}
    top_length: dict[int, float] = {}
    pending: list[Segment] = []
    for s in segments:
        length = float(np.linalg.norm(s.end - s.start))
        direction, middle = _unit(s.end - s.start), (s.start + s.end) / 2
        if _is_wall_step(direction, middle, scene, length, L):
            # Too steep to be any roof edge. "junction" is the label for roof meeting wall:
            # identified, drawn dark, never lit — which is what a notch wall deserves too.
            s.role = "junction"
            continue
        horizontal = _is_plan_horizontal(direction, middle, scene)
        if len(s.planes) == 2:
            if horizontal:
                s.role = "ridge"
                for p in s.planes:
                    top_length[p] = top_length.get(p, 0.0) + length
            elif dormers is not None and s.planes[0] != s.planes[1] and dormers.enclosed(*dormers.child_host(*s.planes)):
                s.role = "rake"
            else:
                s.role = "valley" if _valley(s, labels, L) else "hip"
            continue
        plane = s.planes[0]
        if horizontal:
            if _plane_above(s, labels, plane, L):
                s.role = "eave"
                eave_length[plane] = eave_length.get(plane, 0.0) + length
            else:
                s.role = "ridge"
                top_length[plane] = top_length.get(plane, 0.0) + length
        else:
            pending.append(s)
    for s in pending:
        plane = s.planes[0]
        eave, top = eave_length.get(plane, 0.0), top_length.get(plane, 0.0)
        s.role = "hip" if eave > 0 and top < TOP_TO_EAVE * eave else "rake"


# -- constrained snap (product) ---------------------------------------------------------------------


def _gradient_field(gray: np.ndarray):
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), CONSTRAINED_BLUR)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    minimum = max(float(np.percentile(np.hypot(gx, gy), CONSTRAINED_PERCENTILE)), CONSTRAINED_MIN_STRENGTH)
    return gx, gy, minimum


def constrained_snap(shared: list[Line], junctions: list[Junction], lit: set[int], gray: np.ndarray, band: float) -> int:
    """Translate each lit line along its normal to the strongest parallel edge within the band,
    judged over the whole segment, then re-solve every junction on the translated lines. Returns
    how many lines moved."""
    gx, gy, minimum = _gradient_field(gray)
    height, width = gx.shape
    offsets = np.arange(-int(round(band)), int(round(band)) + 1, dtype=float)
    moved = 0
    for g, line in enumerate(shared):
        if g not in lit:
            continue
        attached = [j.point for j in junctions if g in j.lines]
        if len(attached) >= 2:
            along = sorted(float((p - line.origin) @ line.direction) for p in attached)
            lo, hi = along[0], along[-1]
        else:
            lo = float((line.first - line.origin) @ line.direction)
            hi = float((line.last - line.origin) @ line.direction)
        if hi - lo < 2 * CONSTRAINED_STEP:
            continue
        steps = np.arange(lo, hi, CONSTRAINED_STEP)
        samples = line.origin + np.outer(steps, line.direction)
        n = line.normal
        xs = np.clip(np.rint(samples[:, 0, None] + n[0] * offsets).astype(int), 0, width - 1)
        ys = np.clip(np.rint(samples[:, 1, None] + n[1] * offsets).astype(int), 0, height - 1)
        support = np.abs(gx[ys, xs] * n[0] + gy[ys, xs] * n[1]).mean(axis=0)
        best = int(np.argmax(support - CONSTRAINED_NEARER * np.abs(offsets)))
        if support[best] >= minimum and offsets[best] != 0:
            line.origin = line.origin + n * offsets[best]
            moved += 1
    for junction in junctions:
        if len(junction.lines) < 2:
            if junction.lines:
                junction.point = _project(shared[next(iter(junction.lines))], junction.point)
            continue
        matrix = RIDGE_EPSILON * np.eye(2)
        rhs = RIDGE_EPSILON * junction.point
        for g in junction.lines:
            n = shared[g].normal
            matrix += np.outer(n, n)
            rhs += n * float(n @ shared[g].origin)
        junction.point = np.linalg.solve(matrix, rhs)
    return moved


# -- pitch (step 8, second half) --------------------------------------------------------------------


def _focal_px(image, L: float) -> float | None:
    """f_px = f35 / 36 x L from EXIF FocalLengthIn35mmFilm, when the photo carries it."""
    try:
        exif = image.getexif() if hasattr(image, "getexif") else None
        f35 = exif.get(41989) if exif else None
        if f35 is None and exif is not None:
            f35 = exif.get_ifd(0x8769).get(41989)
    except Exception:
        f35 = None
    return float(f35) / 36.0 * L if f35 else None


def _nearest_standard(angle_deg: float, sigma: float) -> dict:
    """MAP over the standard pitches with the reference's priors and a Gaussian likelihood."""
    best, best_score = None, -np.inf
    for rise, prior in zip(STANDARD_PITCHES, PITCH_PRIORS):
        pitch_deg = float(np.degrees(np.arctan(rise / 12.0)))
        score = np.log(prior) - 0.5 * ((angle_deg - pitch_deg) / sigma) ** 2
        if score > best_score:
            best, best_score = rise, score
    return {"rise": best, "run": 12}


def plane_pitches(segments: list["Segment"], scene: Scene, focal: float | None, L: float) -> dict[int, dict]:
    """Per plane: pitch as seen, calibrated when a focal length exists, and the nearest standard.

    As seen: the length-weighted median angle of the plane's sloped edges against the local
    horizontal, as rise:12 — only valid as seen in the photo. Calibrated: with K, the vertical
    VP and an eave VP give up/along/across; a rake's 3-D direction cos t across + sin t up must
    lie on its image line l, so tan t = -(l K across) / (l K up); a hip uses (along +- across)/
    sqrt 2 and tan t = sqrt 2 tan p. Withheld when a plane's edges disagree by more than
    PITCH_DISAGREE_DEG.
    """
    by_plane: dict[int, list] = {}
    for s in segments:
        if s.role not in ("rake", "hip", "valley"):
            continue
        for plane in set(s.planes):
            by_plane.setdefault(plane, []).append(s)
    calibrated_possible = focal is not None and scene.vertical is not None and scene.horizontal
    K = Kinv = None
    if calibrated_possible:
        K = np.array([[focal, 0, scene.centre[0]], [0, focal, scene.centre[1]], [0, 0, 1.0]])
        Kinv = np.linalg.inv(K)
    out: dict[int, dict] = {}
    for plane, edges in by_plane.items():
        weights, seen, calibrated = [], [], []
        for s in edges:
            direction = _unit(s.end - s.start)
            length = float(np.linalg.norm(s.end - s.start))
            horizontals = _vp_directions(scene, (s.start + s.end) / 2, horizontal_only=True)
            reference = min(horizontals, key=lambda h: _angle_between(direction, h)) if horizontals else np.array([1.0, 0.0])
            apparent = _angle_between(direction, reference)
            if apparent > PITCH_MAX_DEG:
                # A near-vertical edge is a wall step or a cut, not a slope to read a pitch from.
                continue
            weights.append(length)
            seen.append(apparent)
            if calibrated_possible:
                v_vert = scene.vertical[:2] / scene.vertical[2] * scene.scale + scene.centre if abs(scene.vertical[2]) > 1e-9 else None
                if v_vert is None:
                    continue
                up = _unit(Kinv @ np.array([v_vert[0], v_vert[1], 1.0]))
                eave_vp = min(scene.horizontal, key=lambda h: _angle_between(direction, _vp_direction(h, ((s.start + s.end) / 2 - scene.centre) / scene.scale)))
                if abs(eave_vp[2]) < 1e-9:
                    continue
                v_eave = eave_vp[:2] / eave_vp[2] * scene.scale + scene.centre
                along = Kinv @ np.array([v_eave[0], v_eave[1], 1.0])
                along = _unit(along - up * float(along @ up))
                across = _unit(np.cross(up, along))
                if s.role in ("hip", "valley"):
                    across = _unit((along + across) / np.sqrt(2.0))
                a3, b3 = np.array([*s.start, 1.0]), np.array([*s.end, 1.0])
                line = np.cross(a3, b3)
                denominator = float(line @ (K @ up))
                if abs(denominator) < 1e-9:
                    continue
                t = np.arctan(-float(line @ (K @ across)) / denominator)
                if s.role in ("hip", "valley"):
                    t = np.arctan(np.sqrt(2.0) * np.tan(t))
                calibrated.append(abs(float(np.degrees(t))))
        if not weights:
            continue
        order = np.argsort(seen)
        cumulative = np.cumsum(np.array(weights)[order])
        median_seen = float(np.array(seen)[order][int(np.searchsorted(cumulative, cumulative[-1] / 2))])
        entry = {"as_seen": {"degrees": round(median_seen, 1), "rise": round(float(12 * np.tan(np.radians(median_seen))), 1), "run": 12}}
        if calibrated and (max(calibrated) - min(calibrated)) <= PITCH_DISAGREE_DEG:
            median_calibrated = float(np.median(calibrated))
            entry["calibrated"] = {"degrees": round(median_calibrated, 1), "rise": round(float(12 * np.tan(np.radians(median_calibrated))), 1), "run": 12}
            entry["nearest_standard"] = {**_nearest_standard(median_calibrated, PITCH_SIGMA_CALIBRATED_DEG), "basis": "calibrated"}
        else:
            entry["nearest_standard"] = {**_nearest_standard(median_seen, PITCH_SIGMA_APPARENT_DEG), "basis": "as_seen"}
        out[plane] = entry
    return out


# -- entry point --------------------------------------------------------------------------------------


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def _drop_duplicates(segments: list[Segment], L: float) -> list[Segment]:
    """Drop a segment that retraces one already kept with the same role; the longer survives.

    Step 6 merges lines that are already close when the graph is built. The relaxation that
    follows moves them, so two groups can end up drawn along the same roof edge a pixel or two
    apart — one line to look at, two lines drawn, which is what reads as a double roofline.
    Comparing only within a role is what keeps a parapet intact: its eave and the junction above
    it are near-parallel and a few pixels apart, and neither is a copy of the other.
    """
    kept: list[Segment] = []
    for s in sorted(segments, key=lambda x: -float(np.linalg.norm(x.end - x.start))):
        if not any(k.role == s.role and _retraces(k, s, L) for k in kept):
            kept.append(s)
    return kept


def _retraces(long: Segment, short: Segment, L: float) -> bool:
    """Does `short` lie along `long` — parallel, close, and covered by it?"""
    direction, other = _unit(short.end - short.start), _unit(long.end - long.start)
    if _angle_between(direction, other) > DUPLICATE_DEG:
        return False
    normal = np.array([-other[1], other[0]])
    offset = max(abs(float((short.start - long.start) @ normal)), abs(float((short.end - long.start) @ normal)))
    if offset > DUPLICATE_OFFSET_L * L:
        return False
    span = float(np.linalg.norm(short.end - short.start))
    if span < 1e-6:
        return True
    extent = sorted([0.0, float((long.end - long.start) @ other)])
    at = sorted([float((short.start - long.start) @ other), float((short.end - long.start) @ other)])
    return max(0.0, min(extent[1], at[1]) - max(extent[0], at[0])) / span >= DUPLICATE_OVERLAP


def clean_spans(spans: list[tuple], extent: float) -> list[tuple]:
    """Both corrections applied to finished (points, side, edge) spans, whichever engine made them.

    partition_result does this inside itself, where the real vanishing points and line groups are
    to hand. The reference engine returns finished spans and nothing else, so the same two rules
    are repeated here on geometry alone:

    1. An edge far steeper than the roof's own horizontal is a wall step or the wall of a notch
       the mask took around whatever stands in front of the roof. It is dropped. It is not a
       roof edge, nothing downstream needs it, and kept as a dark "junction" it came straight
       back the moment every detected edge was shown — the short blue stubs at each palm trunk
       the client marked as wrong. The horizontal roof-meets-wall junctions from
       _ridges_to_junctions are never this steep and stay.
    2. Two spans of the same role drawn along the same edge are one edge drawn twice. The longer
       survives. Same role only: a thin band's eave and the junction above it are also close and
       near-parallel, and both of those are real.

    The eaves give the horizontal: they are the roof's plan horizontal by definition, which is
    steadier than the image's own horizontal on a photo taken from off to one side.
    """
    if not spans:
        return spans
    reference = _span_horizontal(spans)
    relabelled = [
        (points, side, edge) for points, side, edge in spans
        if not _steep_stub(_angle_between(_span_direction(points), reference), _span_length(points), extent)
    ]
    kept: list[int] = []
    for index in sorted(range(len(relabelled)), key=lambda i: -_span_length(relabelled[i][0])):
        span = relabelled[index]
        if not any(relabelled[k][2] == span[2] and _span_retraces(relabelled[k][0], span[0], extent) for k in kept):
            kept.append(index)
    keep = set(kept)
    return [span for index, span in enumerate(relabelled) if index in keep]


# A bridged line is one straight edge again; the pieces were straight fits of that same edge,
# so anything this far off the chord between the outer ends is a real bend, not a join.
BRIDGE_STRAIGHT_L = 0.006
# Roof pixels grown this much count as covering a gap between two pieces, so a sliver of roof
# left between two trunks does not break the join; the edge itself sits on the mask's boundary.
BRIDGE_ROOF_GROW_PX = 3
# How far off one piece's line the next may sit and still be the same edge. Pieces here are
# straight fits of one edge and agree to a few pixels; hybrid's 1.5% of the image width let two
# eaves a storey apart join across the entry tower when fronds hung over the gap (measured:
# 31 px apart on a 2646 px photo, joined). Same order as the duplicate test.
BRIDGE_OFFSET_L = 0.008


def bridge_through(spans: list[tuple], blockers: np.ndarray | None, width: int, extent: float,
                   roof: np.ndarray | None = None) -> list[tuple]:
    """A roof edge cut by something in front of it becomes one straight line again.

    The roof does not stop at a palm trunk; only the view of it does. Two steps, both driven
    by the detected objects and nothing else, so the arched entry between two garage roofs —
    which no object covers — stays the real gap it is:

    1. occlusion.bridge joins same-role pieces that are collinear and whose gap is covered.
       Cover is the detected objects plus the roof itself: between two trunks a sliver of roof
       often survives, and it is roof, not a gap. Every piece is already a straight fit, so a
       join is straightened to its two outer ends — one edge, one line.
    2. occlusion.extend_behind carries a single remaining end through the object that cut it
       off, to the object's far side, for a band whose end is hidden and has no piece beyond.

    The stretch behind the object is then reported by occlusion.hidden_stretches, never lit.
    """
    if blockers is None:
        return list(spans)
    # Cover is judged the way trust is: within BLOCKER_DILATE_L of the object, not pixel-exact.
    # A palm's crown is ragged — sky shows between the fronds along a one-pixel sample line —
    # yet the eave behind it is hidden all the same (measured on the client's photo: 0.45 exact,
    # 0.71 grown; the real gap at the entry 0.21 -> 0.45, still well under the join threshold).
    cover = cv2.dilate(blockers.astype(np.uint8), _ellipse(BLOCKER_DILATE_L * extent)).astype(bool)
    if roof is not None:
        grown = cv2.dilate(roof.astype(np.uint8), np.ones((2 * BRIDGE_ROOF_GROW_PX + 1,) * 2, np.uint8)).astype(bool)
        cover = cover | grown
    joined = occlusion.bridge(spans, cover, width, max_offset=BRIDGE_OFFSET_L * extent)
    straight = []
    for points, side, edge in joined:
        if len(points) > 2:
            array = np.asarray(points, float)
            chord = array[-1] - array[0]
            norm = np.linalg.norm(chord)
            if norm > 1e-6:
                normal = np.array([-chord[1], chord[0]]) / norm
                if float(np.abs((array - array[0]) @ normal).max()) <= BRIDGE_STRAIGHT_L * extent:
                    points = [points[0], points[-1]]
        straight.append((points, side, edge))
    return occlusion.extend_behind(straight, blockers, extent, roof)


def _span_direction(points) -> np.ndarray:
    return _unit(np.array(points[-1], float) - np.array(points[0], float))


def _span_length(points) -> float:
    array = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def _span_horizontal(spans: list[tuple]) -> np.ndarray:
    """The roof's own horizontal, from its eaves, weighted by length; the image's if it has none."""
    total = np.zeros(2)
    for points, _, edge in spans:
        if edge != "eave":
            continue
        direction = _span_direction(points)
        if direction[0] < 0:
            direction = -direction
        total += direction * _span_length(points)
    return _unit(total) if np.linalg.norm(total) > 1e-6 else np.array([1.0, 0.0])


def _span_retraces(long_points, short_points, extent: float) -> bool:
    other = _span_direction(long_points)
    if _angle_between(_span_direction(short_points), other) > DUPLICATE_DEG:
        return False
    origin = np.array(long_points[0], float)
    normal = np.array([-other[1], other[0]])
    short = np.asarray(short_points, float)
    if float(np.abs((short - origin) @ normal).max()) > DUPLICATE_OFFSET_L * extent:
        return False
    span = _span_length(short_points)
    if span < 1e-6:
        return True
    extent_along = sorted([0.0, float((np.array(long_points[-1], float) - origin) @ other)])
    at = sorted([float((short[0] - origin) @ other), float((short[-1] - origin) @ other)])
    return max(0.0, min(extent_along[1], at[1]) - max(extent_along[0], at[0])) / span >= DUPLICATE_OVERLAP


def partition_result(
    roof_masks: list[np.ndarray],
    width: int,
    blockers: np.ndarray | None = None,
    image=None,
    scores: list[float] | None = None,
    junctions: bool = True,
    constrained: bool = True,
) -> tuple[list[list[tuple]], set, dict]:
    """The lines per plane as (points, side, edge), the keys of the inferred connectors among
    them, and the pitch per plane. `image` is the photo (PIL or RGB array) for foliage,
    vanishing points and EXIF; without it those steps are skipped."""
    height, full_width = roof_masks[0].shape
    factor = min(1.0, WORK_LONG_EDGE / max(height, full_width))
    size = (max(1, round(full_width * factor)), max(1, round(height * factor)))
    masks = [_resize_mask(m, size) if factor < 1 else m.astype(bool) for m in roof_masks]
    L = float(max(size))
    D = float(np.hypot(*size))

    rgb = None
    if image is not None:
        array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        rgb = cv2.resize(array, size, interpolation=cv2.INTER_AREA) if factor < 1 else array
    suspect = foliage_mask(rgb, L)
    blocked = None
    if blockers is not None:
        small = _resize_mask(blockers, size) if factor < 1 else blockers.astype(bool)
        blocked = cv2.dilate(small.astype(np.uint8), _ellipse(BLOCKER_DILATE_L * L)).astype(bool)

    raw_labels = partition(masks, scores or [], L)
    labels = cleanup(raw_labels.copy(), len(masks), L)
    bounds = boundaries(labels, len(masks), L, suspect, blocked)
    # The dormer test reads the map before cleanup filled the dormer's face: there the dormer's
    # eave still borders background inside the outline of host + dormer.
    dormers = Dormers(raw_labels, boundaries(raw_labels, len(masks), L, None))
    polygons = plane_polygons(bounds, L)

    if rgb is not None:
        # Vanishing-point evidence is masked by the colour guess only. Detected objects are
        # kept out of the boundary trust, not out of here: a palm's fronds cover a large share
        # of a photo, and dropping every line segment under them shifted the horizontal
        # vanishing point enough to turn a 6 degree eave into a "rake" (house03, measured).
        segments = lsd_segments(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), suspect)
    else:
        segments = np.zeros((0, 4))
    scene = vanishing_points(segments, (size[1], size[0]))
    if scene.vertical is not None or scene.horizontal:
        apply_priors(polygons, scene)
    else:
        apply_cluster_priors(polygons)

    planes: list[list[tuple]] = [[] for _ in roof_masks]
    if not polygons:
        return planes, set(), {}
    edges, shared, nodes, assigned = build_graph(polygons, scene, L)
    from_graph = graph_segments(edges, shared, nodes, assigned)
    others = interior_creases(polygons, labels, scene, L, D)
    # Inferred connectors: the gaps the corner rule could not close, kept as data.
    others += [Segment(start=e.start, end=e.end, planes=(e.plane,), inferred=True)
               for polygon in polygons for e in polygon if e.inferred]
    assign_roles(from_graph + others, labels, scene, L, dormers)
    if constrained and rgb is not None:
        # Roles are decided on the graph's geometry; the lit lines then move to the fascia.
        lit_length: dict[int, float] = {}
        for s in from_graph:
            share = float(np.linalg.norm(s.end - s.start)) * (1 if s.role in ("eave", "rake", "hip") else -1)
            lit_length[s.key[0]] = lit_length.get(s.key[0], 0.0) + share
        lit = {g for g, share in lit_length.items() if share > 0}
        band = CONSTRAINED_BAND_FRACTION * roof_extent(masks)
        constrained_snap(shared, nodes, lit, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), band)
        # The same segments, ends re-projected from their re-solved junctions: nothing split,
        # nothing dropped, every role kept.
        for s in from_graph:
            g, ja, jb = s.key
            s.start, s.end = _project(shared[g], nodes[ja].point), _project(shared[g], nodes[jb].point)
    # Last, so it judges where the lines finally sit: constrained snap can move two copies of
    # one edge onto each other, and roles are already decided, so the longer survivor keeps its own.
    found = _drop_duplicates(from_graph + others, L)
    pitches = plane_pitches(found, scene, _focal_px(image, L) if image is not None else None, L)

    sides = [roofline.plane_side(mask, width) for mask in roof_masks]
    inferred: set = set()
    for s in found:
        if float(np.linalg.norm(s.end - s.start)) < max(1.0, MIN_SEGMENT_L * L):
            continue
        start, end = s.start / factor, s.end / factor
        points = [[int(round(start[0])), int(round(start[1]))], [int(round(end[0])), int(round(end[1]))]]
        if points[0] == points[1]:
            continue
        if points[0][0] > points[1][0]:
            points = points[::-1]
        owner = s.planes[0]
        if len(s.planes) == 2 and s.planes[0] != s.planes[1]:
            a, b = sides[s.planes[0]], sides[s.planes[1]]
            side = "front" if "front" in (a, b) else (a if a == b else roofline._classify(np.array(points, float), width))
        else:
            side = sides[owner]
        planes[owner].append((points, side, s.role))
        if s.inferred:
            inferred.add(tuple(map(tuple, points)))
    if junctions:
        # Product rule kept from the hybrid path (Settings.partition_junctions): a lower roof's
        # top edge that runs into the wall of the storey above is relabelled "junction". Only a
        # ridge is offered to the test — the role table has already found the horizontal top
        # edges, and on partition output that is what a wall junction looks like. Offering hips
        # and rakes as well is what used to strip whole hips off the lower roof (3 lines, 957 px
        # on the test house) whenever the upper storey happened to rise above them.
        planes = _ridges_to_junctions(planes, roof_masks, width)
    return planes, inferred, pitches


def _ridges_to_junctions(planes: list[list[tuple]], roof_masks: list[np.ndarray], width: int) -> list[list[tuple]]:
    ridges_only = [[span for span in spans if span[2] == "ridge"] for spans in planes]
    marked = roofline.mark_junctions(ridges_only, roof_masks, width)
    out = []
    for spans, relabelled in zip(planes, marked):
        replaced = iter(relabelled)
        out.append([next(replaced) if span[2] == "ridge" else span for span in spans])
    return out


def partition_lines(roof_masks, width, blockers=None, image=None, scores=None) -> list[list[tuple]]:
    """Per plane, its lines as (points, side, edge) — the structure roof_lines assembles."""
    return partition_result(roof_masks, width, blockers, image, scores)[0]
