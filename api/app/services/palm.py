import numpy as np

MIN_ROWS = 12
WIDEN_FACTOR = 2.2
LOWER_SAMPLE = 0.35
MAX_TRUNK_FRACTION = 0.45


def row_extent(mask: np.ndarray) -> np.ndarray:
    extent = np.zeros(mask.shape[0], dtype=int)
    rows = np.where(mask.any(axis=1))[0]
    for y in rows:
        xs = np.where(mask[y])[0]
        extent[y] = xs.max() - xs.min() + 1
    return extent


def split(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
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
    frond = mask.copy()
    if husk is None:
        frond[:, :] = False
    else:
        trunk[: husk + 1, :] = False
        frond[husk + 1 :, :] = False

    if not trunk.any():
        return None
    return trunk, frond
