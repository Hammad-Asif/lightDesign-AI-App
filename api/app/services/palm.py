import cv2
import numpy as np

MIN_ROWS = 12
# The crown starts where the palm is this many trunk widths across, held for SUSTAIN_SHARE of its
# height: lower than the old 2.2 widths, which left a long stretch of crown base lit as trunk,
# and held so one stray leaf beside the trunk does not cut the trunk short.
WIDEN_FACTOR = 1.7
SUSTAIN_SHARE = 0.02
LOWER_SAMPLE = 0.35
MAX_TRUNK_FRACTION = 0.45

# The collar is the band of trunk just below the fronds, lit as a zone of its own (a red collar
# between a warm white trunk and green fronds). Its height is measured in trunk widths, so it
# scales with the palm, and capped so a short trunk keeps most of its length as trunk. Kept short:
# on real palms it is the boots under the crown, not a stretch of the trunk.
COLLAR_HEIGHT = 0.8
COLLAR_MAX_SHARE = 0.12


def row_extent(mask: np.ndarray) -> np.ndarray:
    extent = np.zeros(mask.shape[0], dtype=int)
    rows = np.where(mask.any(axis=1))[0]
    for y in rows:
        xs = np.where(mask[y])[0]
        extent[y] = xs.max() - xs.min() + 1
    return extent


def split(
    mask: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """A palm's mask as three lighting zones: (trunk, collar, fronds).

    The fronds start at the husk, the first row up from the base where the palm stays well wider
    than its trunk. The collar is the band directly beneath that. Without a husk there is no crown
    to measure from, so collar and fronds come back empty and the whole mask is trunk.
    """
    extent = row_extent(mask)
    rows = np.where(extent > 0)[0]
    if rows.size < MIN_ROWS:
        return None

    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    lower = extent[max(bottom - int(height * LOWER_SAMPLE), top) : bottom + 1]
    lower = lower[lower > 0]
    if lower.size < 3:
        return None

    trunk_width = float(np.median(lower))
    box_width = max(box[2] - box[0], 1)
    if trunk_width <= 0 or trunk_width > MAX_TRUNK_FRACTION * box_width:
        return None

    limit = trunk_width * WIDEN_FACTOR
    sustain = max(2, round(SUSTAIN_SHARE * height))
    husk = None
    for y in range(bottom, top - 1, -1):
        above = extent[max(top, y - sustain + 1) : y + 1]
        if extent[y] > limit and (above > limit).all():
            husk = y
            break

    trunk = mask.copy()
    collar = np.zeros_like(mask)
    frond = mask.copy()
    if husk is None:
        frond[:, :] = False
    else:
        trunk[: husk + 1, :] = False
        frond[husk + 1 :, :] = False
        band = min(round(COLLAR_HEIGHT * trunk_width), int(COLLAR_MAX_SHARE * (bottom - husk)))
        if band > 0:
            collar[husk + 1 : husk + 1 + band, :] = trunk[husk + 1 : husk + 1 + band, :]
            trunk[husk + 1 : husk + 1 + band, :] = False

    if not trunk.any():
        return None
    return trunk, collar, frond


# -- fronds as lines -------------------------------------------------------------------------------
#
# Lights on fronds run along each frond, from the crown out to its tip, as an installer strings them.
# A frond mask is a crown-shaped blob, so the fronds are read from its outline: each tip is a peak
# in how far the mask reaches from the crown in that direction.

RAY_STEP_DEG = 2.0
RAY_GAP_SHARE = 0.03       # of the crown's size: a gap this short between leaflets is still frond
TIP_MIN_SHARE = 0.35       # a tip reaches at least this share of the farthest one
TIP_MIN_DEG = 14.0         # two fronds' tips are at least this far apart around the crown
MAX_FRONDS = 16
DOWN_CONE_DEG = 25.0       # straight down from the crown is the trunk, not a frond
ARCH = 0.22                # a sideways or drooping frond rises from the crown before it falls
INSIDE_SHARE = 0.75        # a traced frond keeps at least this share of itself on the mask


def crown_of(frond: np.ndarray, stem: np.ndarray | None = None) -> tuple[float, float] | None:
    """Where the fronds leave the palm: the top of the trunk (or collar) under them when it is
    known, else the thickest part of the crown."""
    if not frond.any():
        return None
    if stem is not None and stem.any():
        ys, xs = np.nonzero(stem)
        top = ys.min()
        near = ys <= top + max(2, 0.03 * (ys.max() - top))
        cx = float(xs[near].mean())
        x0, x1 = np.nonzero(frond.any(axis=0))[0][[0, -1]]
        if x0 - 2 <= cx <= x1 + 2:
            return cx, float(top)
    distance = cv2.distanceTransform(frond.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return float(x), float(y)


def _bezier(a: np.ndarray, c: np.ndarray, b: np.ndarray, count: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def frond_paths(frond: np.ndarray, stem: np.ndarray | None = None) -> list[np.ndarray]:
    """Each frond as a path from the crown to its tip, in mask pixels; empty when none is found."""
    crown = crown_of(frond, stem)
    if crown is None:
        return []
    height, width = frond.shape
    ys, xs = np.nonzero(frond)
    size = float(max(np.ptp(xs), np.ptp(ys), 1))
    closing = max(1, round(0.01 * size))
    solid = cv2.morphologyEx(frond.astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closing + 1, 2 * closing + 1))) > 0
    gap = max(2.0, RAY_GAP_SHARE * size)
    centre = np.array(crown, float)

    angles = np.deg2rad(np.arange(0.0, 360.0, RAY_STEP_DEG))
    reach = np.zeros(len(angles))
    radii = np.arange(1.0, 1.5 * size, 1.0)
    for index, angle in enumerate(angles):
        px = np.rint(centre[0] + radii * np.cos(angle)).astype(int)
        py = np.rint(centre[1] + radii * np.sin(angle)).astype(int)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        on = np.zeros(len(radii), bool)
        on[inside] = solid[py[inside], px[inside]]
        hit = radii[on]
        if not hit.size:
            continue
        # Out from the crown until a gap longer than `gap`: the last hit before it is the reach.
        steps = np.diff(np.concatenate(([0.0], hit))) - 1.0
        broken = np.flatnonzero(steps > gap)
        if not broken.size:
            reach[index] = hit[-1]
        elif broken[0] > 0:
            reach[index] = hit[broken[0] - 1]
    if not reach.any():
        return []

    # Circular smoothing, then peaks: a frond's tip is farther out than the outline either side.
    smooth = (np.roll(reach, 1) + reach + np.roll(reach, -1)) / 3.0
    floor = TIP_MIN_SHARE * smooth.max()
    down = np.abs(np.rad2deg(angles) - 90.0) <= DOWN_CONE_DEG if stem is not None and stem.any() else np.zeros(len(angles), bool)
    peaks = [i for i in range(len(angles))
             if smooth[i] >= floor and not down[i]
             and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[(i + 1) % len(angles)]]
    peaks.sort(key=lambda i: -smooth[i])
    chosen: list[int] = []
    separation = TIP_MIN_DEG / RAY_STEP_DEG
    for i in peaks:
        if all(min(abs(i - j), len(angles) - abs(i - j)) >= separation for j in chosen):
            chosen.append(i)
        if len(chosen) >= MAX_FRONDS:
            break

    paths = []
    for i in sorted(chosen):
        tip = centre + reach[i] * np.array([np.cos(angles[i]), np.sin(angles[i])])
        length = float(np.linalg.norm(tip - centre))
        if length < 4:
            continue
        count = max(8, int(length / 2))
        best = None
        # Arched first (a frond rises before it droops), flatter, then straight: the first that
        # stays on the mask. An upright frond is straight.
        rising = tip[1] - centre[1] < -0.6 * length
        for arch in ((0.0,) if rising else (ARCH, ARCH / 2, 0.0)):
            control = (centre + tip) / 2 - np.array([0.0, arch * length])
            path = _bezier(centre, control, tip, count)
            px = np.clip(np.rint(path[:, 0]).astype(int), 0, width - 1)
            py = np.clip(np.rint(path[:, 1]).astype(int), 0, height - 1)
            if float(solid[py, px].mean()) >= INSIDE_SHARE:
                best = path
                break
        paths.append(best if best is not None else _bezier(centre, (centre + tip) / 2, tip, count))
    return paths
