"""Rooflines that pass behind something in front of the house: a palm, a tree, a car.

The object stays in the photo and gets no lights. The roofline behind it is still real and
still has to be measured, so a line that an object breaks in two is joined back up, and the
stretch behind the object is recorded as hidden rather than dropped.

Hidden stretches are always derived from the masks of blocking objects, never stored as a
fact about the line. Correcting an object in the editor — painting it larger, erasing it,
deleting it — is therefore enough to correct every line that passes behind it.

A join needs a detected object covering the gap. Joining on geometry alone — two pieces of
one roof plane that line up — was tried first and measured on the client's photos: of three
joins, the two with a palm in front were right, and the one with nothing in front invented a
hidden stretch across a roof that simply steps down. The object is the evidence.
"""

import cv2
import numpy as np

from app.services import roofline

# Kinds that can stand between the camera and the house.
BLOCKER_KINDS = {
    "obstruction", "palm_trunk", "palm_collar", "palm_frond", "tree_canopy", "hedge", "shrub",
    "topiary",
}

# Of those, the ones whose own outline is straight enough that a line fitter takes it for a roof
# edge: a trunk, a post, a car. Only these are kept out of the boundary fit — partition's colour
# guess plus its straightness test handle a ragged crown, and forcing every frond out of the fit
# loses real roof that is plainly visible between them (measured: house05 recall 63.5 -> 57.8).
# Everything in BLOCKER_KINDS still counts for hiding a line and for joining it back up.
SOLID_KINDS = {"obstruction", "palm_trunk"}

MAX_GAP_FRACTION = 0.25        # of image width
MAX_OFFSET_FRACTION = 0.015    # how far the far piece may sit off the near piece's line
ANGLE_TOLERANCE_DEG = 12.0     # between the two pieces' directions
JOIN_COVER = 0.6               # share of a gap a detected object must cover
SAMPLE_STEP = 2.0              # pixels between samples along a line
MIN_HIDDEN_FRACTION = 0.004    # of image width; shorter hidden stretches are noise
# Carrying a single end through an object (extend_behind): the object must begin this close
# past the end, and may be at most this wide along the line — a trunk, a post, a car; not a
# hedge running the length of the house.
EXTEND_REACH_FRACTION = 0.02   # of the roof's extent
EXTEND_MAX_FRACTION = 0.15     # of the roof's extent


def blocker_union(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray | None:
    """Every blocking object as one mask, grown a pixel so a line on its edge still counts."""
    if not masks:
        return None
    union = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        union |= mask.astype(np.uint8)
    return cv2.dilate(union, np.ones((3, 3), np.uint8)).astype(bool)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _rightward(points: np.ndarray) -> bool:
    return (
        len(points) >= 2
        and bool(np.all(np.diff(points[:, 0]) >= 0))
        and points[-1][0] > points[0][0]
    )


def _samples(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points every SAMPLE_STEP pixels along a polyline, with their distance from its start."""
    coords: list[np.ndarray] = []
    distances: list[float] = []
    walked = 0.0
    for a, b in zip(points[:-1].astype(float), points[1:].astype(float)):
        length = float(np.linalg.norm(b - a))
        if length == 0:
            continue
        for t in np.arange(0.0, length, SAMPLE_STEP):
            coords.append(a + (b - a) * (t / length))
            distances.append(walked + t)
        walked += length
    coords.append(points[-1].astype(float))
    distances.append(walked)
    return np.asarray(coords), np.asarray(distances)


def _on_mask(mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    xs = np.clip(np.round(coords[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(coords[:, 1]).astype(int), 0, height - 1)
    return mask[ys, xs]


def _coverage(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    coords, _ = _samples(np.stack([a, b]).astype(float))
    return float(_on_mask(mask, coords).mean())


def _gap_if_continues(
    left: np.ndarray, right: np.ndarray, blockers: np.ndarray, width: int, max_offset: float | None = None
) -> float | None:
    """The gap length if `right` carries on from `left` behind a detected object, else None.

    `max_offset` is how far off the near piece's line the far piece may sit. Hybrid's traced
    polylines wobble and get MAX_OFFSET_FRACTION of the width; straight fits of one edge agree
    to a few pixels and pass something tighter, so two roofs a storey apart never join.
    """
    a, b = left[-1].astype(float), right[0].astype(float)
    gap = b - a
    distance = float(np.linalg.norm(gap))
    if gap[0] <= 0 or distance > MAX_GAP_FRACTION * width:
        return None

    heading = _unit((left[-1] - left[-2]).astype(float))
    onward = _unit((right[1] - right[0]).astype(float))
    turn = np.degrees(np.arccos(np.clip(float(np.dot(heading, onward)), -1.0, 1.0)))
    if turn > ANGLE_TOLERANCE_DEG:
        return None

    normal = np.array([-heading[1], heading[0]])
    limit = MAX_OFFSET_FRACTION * width if max_offset is None else max_offset
    if abs(float(np.dot(gap, normal))) > limit:
        return None

    if _coverage(blockers, a, b) < JOIN_COVER:
        return None
    return distance


def bridge(spans: list[tuple], blockers: np.ndarray | None, width: int, max_offset: float | None = None) -> list[tuple]:
    """Join roofline pieces that continue behind a detected object.

    `spans` are (points, side, edge). Pieces are joined end to end, shortest gap first. A
    joined line keeps its pieces' side when they agree — side belongs to the roof plane, not
    to a line's slope — and is classified afresh only when they do not.
    """
    if blockers is None:
        return list(spans)

    pieces = [
        {"points": np.asarray(points), "side": side, "edge": edge, "joined": False}
        for points, side, edge in spans
    ]
    while True:
        best = None
        for i, left in enumerate(pieces):
            if not _rightward(left["points"]):
                continue
            for j, right in enumerate(pieces):
                if i == j or right["edge"] != left["edge"] or not _rightward(right["points"]):
                    continue
                gap = _gap_if_continues(left["points"], right["points"], blockers, width, max_offset)
                if gap is not None and (best is None or gap < best[0]):
                    best = (gap, i, j)
        if best is None:
            break

        _, i, j = best
        same = pieces[i]["side"] is not None and pieces[i]["side"] == pieces[j]["side"]
        joined = {
            "points": np.vstack([pieces[i]["points"], pieces[j]["points"]]),
            "side": pieces[i]["side"] if same else None,
            "edge": pieces[i]["edge"],
            "joined": True,
        }
        pieces = [piece for k, piece in enumerate(pieces) if k not in (i, j)] + [joined]

    return [
        (
            [[int(x), int(y)] for x, y in piece["points"]],
            piece["side"] if piece["side"] is not None else roofline._classify(piece["points"], width),
            piece["edge"],
        )
        for piece in pieces
    ]


def extend_behind(spans: list[tuple], blockers: np.ndarray | None, extent: float,
                  roof: np.ndarray | None = None) -> list[tuple]:
    """Carry a line's end through the object that cut it off, as far as the roof is known to go.

    bridge needs a piece on each side of the object. A roof band whose END an object hides has
    only one, so the line stops at the near edge of the palm trunk as if the roof did. Each end
    is walked onward along the line's own direction: if a detected object begins within
    EXTEND_REACH of the end and the walk comes out its far side within EXTEND_MAX, the end moves
    to where it comes out.

    `roof` is what makes this a measurement rather than a guess: the end only moves if there is
    roof at the far side. How far the roof really runs behind an object cannot be seen, so
    nothing is invented past where the masks show roof — a hip that stops at its apex under a
    frond stays where it is. Without `roof` no end moves at all.
    """
    if blockers is None or roof is None:
        return list(spans)
    height, width = blockers.shape
    reach, limit = EXTEND_REACH_FRACTION * extent, EXTEND_MAX_FRACTION * extent
    confirmed = cv2.dilate(roof.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out = []
    for points, side, edge in spans:
        array = np.asarray(points, dtype=float)
        if len(array) < 2:
            out.append((points, side, edge))
            continue
        first = _carry(array[0], _unit(array[0] - array[1]), blockers, reach, limit, confirmed)
        last = _carry(array[-1], _unit(array[-1] - array[-2]), blockers, reach, limit, confirmed)
        if first is None and last is None:
            out.append((points, side, edge))
            continue
        moved = array.copy()
        if first is not None:
            moved[0] = first
        if last is not None:
            moved[-1] = last
        moved[:, 0] = np.clip(moved[:, 0], 0, width - 1)
        moved[:, 1] = np.clip(moved[:, 1], 0, height - 1)
        out.append(([[int(round(x)), int(round(y))] for x, y in moved], side, edge))
    return out


def _carry(end: np.ndarray, onward: np.ndarray, blockers: np.ndarray, reach: float, limit: float,
           confirmed: np.ndarray) -> np.ndarray | None:
    """Where a walk from `end` along `onward` leaves the object it runs into, or None.

    `reach` bounds how far past the end the object may begin; `limit` bounds the object's width
    along the line, counted from where the walk enters it. The point it comes out at has to be
    roof (`confirmed`), or the end does not move: past the object the roof is not visible, and
    a line is only carried to where the masks say there is roof to carry it to.
    """
    if not np.any(onward):
        return None
    height, width = blockers.shape
    entered = None
    step = 0.0
    while True:
        point = end + onward * step
        x, y = int(round(point[0])), int(round(point[1]))
        if not (0 <= x < width and 0 <= y < height):
            return None
        inside = bool(blockers[y, x])
        if entered is None:
            if inside:
                entered = step
            elif step > reach:
                return None
        elif not inside:
            return point if confirmed[y, x] else None
        elif step - entered > limit:
            return None
        step += SAMPLE_STEP


def hidden_stretches(
    points: list[list[int]], blockers: np.ndarray | None, width: int
) -> list[list[float]]:
    """[start, end] distances along a line, in pixels, where it passes behind an object."""
    array = np.asarray(points, dtype=float)
    if blockers is None or len(array) < 2:
        return []
    coords, distances = _samples(array)
    covered = _on_mask(blockers, coords)

    stretches: list[list[float]] = []
    start = None
    for inside, distance in zip(covered, distances):
        if inside and start is None:
            start = float(distance)
        elif not inside and start is not None:
            stretches.append([start, float(distance)])
            start = None
    if start is not None:
        stretches.append([start, float(distances[-1])])

    minimum = MIN_HIDDEN_FRACTION * width
    return [[round(s, 1), round(e, 1)] for s, e in stretches if e - s >= minimum]
