import numpy as np

MIN_ROWS = 12
WIDEN_FACTOR = 2.2
LOWER_SAMPLE = 0.35
MAX_TRUNK_FRACTION = 0.45

# The collar is the band of trunk just below the fronds, lit as a zone of its own (a red collar
# between a warm white trunk and green fronds). Its height is measured in trunk widths, so it
# scales with the palm, and capped so a short trunk keeps most of its length as trunk.
COLLAR_HEIGHT = 1.5
COLLAR_MAX_SHARE = 0.25


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

    The fronds start at the husk, the first row up from the base where the palm widens well past
    its trunk. The collar is the band directly beneath that. Without a husk there is no crown to
    measure from, so collar and fronds come back empty and the whole mask is trunk.
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
    husk = None
    for y in range(bottom, top - 1, -1):
        if extent[y] > limit:
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
