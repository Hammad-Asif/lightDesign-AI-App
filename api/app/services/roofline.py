import cv2
import numpy as np

ANGLE_SPLIT_DEG = 25.0
MIN_SPAN_FRACTION = 0.06
FRONT_SLOPE_MAX = 0.18
GAP_FRACTION = 0.006
JUMP_FRACTION = 0.02

# Hybrid edge extraction. A profile span has to reach this far across the plane's own
# width to be trusted; below that the outline polygon is used instead.
PROFILE_MIN_FRAC = 0.50
POLY_EPSILON = 0.015
POLY_MIN_LEN = 0.03

# Simplifying the outline can cut a square corner with a diagonal across whatever stands
# beside the plane — on the customer photo, across the stucco of an entry tower. A side that
# mostly misses its own plane is traced along the mask outline between the same two corners.
POLY_ON_MASK_MIN = 0.50
POLY_ON_MASK_TOLERANCE = 4    # px; a straight chord along a jagged outline lands either side of it

# Staged, off: minimum lengths measured against the plane's own width instead of the image's,
# so a small plane or a piece between palm fronds is not dropped for being short next to the
# whole photo. Recovers edges but adds short lines to clean up; not decided yet.
PLANE_MIN_SPAN = False
MIN_SPAN_FLOOR_PX = 40.0

# Adjacency-based edge labelling. What sits just outside a plane's boundary says what that
# boundary is: wall below it means an eave, sky beyond it means a ridge or a rake. Slope
# alone cannot tell a rake from a ridge reliably, and it never distinguished either from an
# eave — which is why the top profile used to be called "ridge" even on gable rakes.
OUTWARD_FRACTION = 0.012   # how far outside the plane to probe, as a share of image width
ADJACENCY_MIN = 0.50       # share of probes that must agree before adjacency decides
RIDGE_SLOPE_MAX = 0.18     # a sky-facing edge flatter than this, high on the plane, is a ridge
# A low edge sloping less than this is an eave the camera angle tilted. Steeper, it could be
# the foot of a rake, so it stays unidentified.
EAVE_MAX_DEG = 15.0

# A ridge or hip is where two planes MEET, so it only exists between two abutting masks.
# Dilation grows each mask by kernel//2, so closing a gap of g needs a kernel of g+1.
# Measured on 176 non-overlapping pairs from the eval set: closest approach is sharply
# bimodal (p10 2px, p25 55px), and the qualifying count is flat from 4px to 12px
# (24, 26, 27 pairs) before jumping at 20px, where unrelated structures start merging.
SEAM_DILATE_FRACTION = 0.006
SEAM_MAX_IOU = 0.10        # above this the two masks are duplicates, not neighbours
SEAM_MIN_LEN = 0.03        # of image width

# A seam sloping like a roof corner is a hip; a level one is a ridge. Steeper than HIP_MAX_DEG
# it is a wall stepping between two roofs at different depths, not a roof line — measured, 3
# such seams at 72-80 deg — so it stays "ridge" and an outline request leaves it dark.
HIP_MIN_DEG = 15.0
HIP_MAX_DEG = 60.0

# A plane faces the front when its eave runs level. Checked two ways, because a gable front's
# mask is often only the thin bands along its rakes: the chord between the lowest points at
# the plane's two ends, or a level eave run covering this share of the plane's width.
FRONT_EAVE_SHARE = 0.50

# Roof-to-wall junctions. The slope rule calls every sloping top edge a rake, but a lower roof
# whose top edge runs into the wall of the storey above has nothing there to hang lights on.
# From the front the difference is what lies above the edge: sky above a real hip or rake;
# above a junction, a band of wall and then the upper roof. Masks that overlap count as one
# plane, or a plane's own duplicate would be read as the roof above it.
JUNCTION_SNAP_FRACTION = 0.008   # of image width: how near a sample must be to its plane's top
JUNCTION_GAP_FRACTION = 0.004    # of image width: wall needed between the edge and the roof above
JUNCTION_REACH = 0.25            # of image height: how far up to look for that roof
JUNCTION_MIN_SHARE = 0.50        # of the span's samples
SAME_PLANE_CONTAINMENT = 0.50    # overlap as a share of the smaller mask

# Which label survives when two near-duplicate spans disagree: a useful, known edge beats
# an unknown one, so dedupe never drops a labelled span in favour of an unlabelled twin.
EDGE_ORDER = {"eave": 0, "rake": 1, "hip": 2, "ridge": 3, "junction": 4, "unknown": 5}


def top_profile(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    occupied = binary.any(axis=0)
    if not occupied.any():
        return np.empty((0, 2), dtype=np.int32)
    first = binary.argmax(axis=0)
    columns = np.where(occupied)[0]
    return np.stack([columns, first[columns]], axis=1).astype(np.int32)


def bottom_profile(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    occupied = binary.any(axis=0)
    if not occupied.any():
        return np.empty((0, 2), dtype=np.int32)
    last = mask.shape[0] - 1 - binary[::-1].argmax(axis=0)
    columns = np.where(occupied)[0]
    return np.stack([columns, last[columns]], axis=1).astype(np.int32)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _classify(span: np.ndarray, width: int) -> str:
    start, end = span[0], span[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dy) / max(abs(dx), 1.0) < FRONT_SLOPE_MAX:
        return "front"
    return "left" if (start[0] + end[0]) / 2 < width / 2 else "right"


def _runs(profile: np.ndarray, width: int, height: int) -> list[np.ndarray]:
    gap_limit = max(2.0, GAP_FRACTION * width)
    jump_limit = max(3.0, JUMP_FRACTION * height)
    runs: list[np.ndarray] = []
    current = [profile[0]]
    for previous, point in zip(profile, profile[1:]):
        broken = (
            point[0] - previous[0] > gap_limit
            or abs(int(point[1]) - int(previous[1])) > jump_limit
        )
        if broken:
            runs.append(np.array(current))
            current = [point]
        else:
            current.append(point)
    runs.append(np.array(current))
    return runs


def _split_run(run: np.ndarray, width: int) -> list[np.ndarray]:
    epsilon = max(2.0, 0.004 * width)
    simplified = cv2.approxPolyDP(run.reshape(-1, 1, 2), epsilon, False).reshape(-1, 2)
    if len(simplified) < 2:
        return []

    spans: list[np.ndarray] = []
    current = [simplified[0]]
    limit = np.deg2rad(ANGLE_SPLIT_DEG)
    for i in range(1, len(simplified)):
        current.append(simplified[i])
        if i < len(simplified) - 1:
            delta = _angle(simplified[i], simplified[i + 1]) - _angle(
                simplified[i - 1], simplified[i]
            )
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            if abs(delta) > limit:
                spans.append(np.array(current))
                current = [simplified[i]]
    spans.append(np.array(current))
    return spans


def extract_spans(mask: np.ndarray, edge: str = "eave") -> list[tuple[list[list[int]], str]]:
    profile = bottom_profile(mask) if edge == "eave" else top_profile(mask)
    if len(profile) < 2:
        return []

    height, width = mask.shape
    minimum = _minimum(MIN_SPAN_FRACTION, mask, width)
    result: list[tuple[list[list[int]], str]] = []

    for run in _runs(profile, width, height):
        if len(run) < 2 or _length(run) < minimum:
            continue
        for span in _split_run(run, width):
            if len(span) >= 2 and _length(span) >= minimum:
                result.append(
                    ([[int(x), int(y)] for x, y in span], _classify(span, width))
                )
    return result


def _plane_width(mask: np.ndarray) -> int:
    columns = np.where(mask.any(axis=0))[0]
    return int(columns[-1] - columns[0] + 1) if columns.size else 0


def _minimum(fraction: float, mask: np.ndarray, image_width: int) -> float:
    """Shortest line kept: a share of the image width, or of the plane's width when staged."""
    if PLANE_MIN_SPAN:
        return max(MIN_SPAN_FLOOR_PX, fraction * _plane_width(mask))
    return fraction * image_width


def _along(points, step: float = 2.0) -> np.ndarray:
    """Points every `step` pixels along a polyline, ends included."""
    array = np.asarray(points, dtype=float)
    coords = []
    for a, b in zip(array[:-1], array[1:]):
        length = float(np.linalg.norm(b - a))
        for travelled in np.arange(0.0, length, step):
            coords.append(a + (b - a) * (travelled / length))
    coords.append(array[-1])
    return np.asarray(coords)


def _on_share(mask: np.ndarray, points) -> float:
    height, width = mask.shape
    coords = np.rint(_along(points)).astype(int)
    xs = np.clip(coords[:, 0], 0, width - 1)
    ys = np.clip(coords[:, 1], 0, height - 1)
    return float(mask[ys, xs].mean())


def _corner_positions(trace: np.ndarray, corners: np.ndarray) -> list[int]:
    """Where each simplified corner sits on the full outline, in outline order.

    Simplification keeps original outline points, so every corner is found exactly. An outline
    can pass the same pixel twice where the mask is one pixel thin; the occurrence that follows
    the previous corner is the one simplification kept.
    """
    positions: list[int] = []
    for corner in corners:
        matches = np.flatnonzero((trace == corner).all(axis=1))
        if not positions:
            positions.append(int(matches[0]))
        else:
            previous = positions[-1]
            positions.append(int(min(matches, key=lambda i: (i - previous) % len(trace))))
    return positions


def _outline_between(trace: np.ndarray, start: int, end: int, image_width: int) -> list[list[int]]:
    """The mask outline from one corner to the next, smoothed like a profile run."""
    steps = (end - start) % len(trace)
    path = trace[(start + np.arange(steps + 1)) % len(trace)]
    epsilon = max(2.0, 0.004 * image_width)
    return cv2.approxPolyDP(path.reshape(-1, 1, 2), epsilon, False).reshape(-1, 2).tolist()


def _polygon_spans(mask: np.ndarray, image_width: int) -> list[tuple]:
    """Every side of the plane's outline, not just its top and bottom profiles."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return []
    outline = max(contours, key=cv2.contourArea)
    corners = cv2.approxPolyDP(
        outline, POLY_EPSILON * cv2.arcLength(outline, True), True
    ).reshape(-1, 2)
    trace = outline.reshape(-1, 2)
    positions = _corner_positions(trace, corners)
    size = 2 * POLY_ON_MASK_TOLERANCE + 1
    near = cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    minimum = _minimum(POLY_MIN_LEN, mask, image_width)

    # Side comes from the segment's own geometry and the edge type is decided later by
    # adjacency. Neither is stamped here: the old ("front", "eave") literal was fabricated
    # and every polygon-path span inherited it regardless of where it actually sat.
    spans = []
    for index, (a, b) in enumerate(zip(corners, np.roll(corners, -1, axis=0))):
        if float(np.linalg.norm(b.astype(float) - a)) < minimum:
            continue
        points = [a.tolist(), b.tolist()]
        if _on_share(near, points) < POLY_ON_MASK_MIN:
            following = positions[(index + 1) % len(corners)]
            points = _outline_between(trace, positions[index], following, image_width)
        spans.append((points, _classify(np.stack([a, b]), image_width), "unknown"))
    return spans


def _outward_probes(points, plane: np.ndarray, image_width: int) -> np.ndarray:
    """Points just outside the plane along a boundary run.

    For each sample the two perpendicular offsets are tested and the one that leaves the
    plane is kept. Where both sides are off the plane the sample is dropped rather than
    guessed, which is what keeps a thin sliver of mask from voting on its own label.
    """
    array = np.asarray(points, dtype=float)
    if len(array) < 2:
        return np.empty((0, 2), dtype=int)
    height, width = plane.shape
    distance = max(4.0, OUTWARD_FRACTION * image_width)
    step = max(2.0, distance / 2)

    probes: list[tuple[int, int]] = []
    for a, b in zip(array[:-1], array[1:]):
        segment = b - a
        length = float(np.linalg.norm(segment))
        if length == 0:
            continue
        normal = np.array([-segment[1], segment[0]]) / length
        for travelled in np.arange(0.0, length, step):
            origin = a + segment * (travelled / length)
            sides = []
            for sign in (1.0, -1.0):
                probe = origin + normal * distance * sign
                x, y = int(round(probe[0])), int(round(probe[1]))
                if 0 <= x < width and 0 <= y < height:
                    sides.append((x, y, bool(plane[y, x])))
            outside = [(x, y) for x, y, on_plane in sides if not on_plane]
            if len(outside) == 1:
                probes.append(outside[0])
    return np.asarray(probes, dtype=int) if probes else np.empty((0, 2), dtype=int)


def _sky_edge(points, plane: np.ndarray) -> str:
    """Split a sky-facing edge: sloping is a rake, flat and high is a ridge."""
    array = np.asarray(points, dtype=float)
    dx = abs(float(array[-1][0] - array[0][0]))
    dy = abs(float(array[-1][1] - array[0][1]))
    rows = np.where(plane.any(axis=1))[0]
    if rows.size == 0:
        return "unknown"
    high = float(array[:, 1].mean()) <= (float(rows[0]) + float(rows[-1])) / 2.0
    if dy / max(dx, 1.0) >= RIDGE_SLOPE_MAX:
        return "rake"
    # A flat edge low on the plane is an eave seen against sky, not a ridge.
    return "ridge" if high else "eave"


def _adjacent_edge(points, plane: np.ndarray, context, image_width: int) -> str | None:
    """eave / ridge / rake from what sits outside the boundary, or None if nothing decides.

    `context` is (sky, wall). The wall mask must already have the roof subtracted: the
    "facade" concept overlaps the roof by ~93% of roof area, so an unsubtracted mask votes
    "wall" for every boundary and the test becomes meaningless.
    """
    if context is None:
        return None
    sky, wall = context
    probes = _outward_probes(points, plane, image_width)
    if len(probes) == 0:
        return None
    xs, ys = probes[:, 0], probes[:, 1]
    sky_share = float(sky[ys, xs].mean()) if sky is not None and sky.any() else 0.0
    wall_share = float(wall[ys, xs].mean()) if wall is not None and wall.any() else 0.0

    if wall_share >= ADJACENCY_MIN and wall_share > sky_share:
        return "eave"
    if sky_share >= ADJACENCY_MIN and sky_share > wall_share:
        return _sky_edge(points, plane)
    return None


def _slope_edge(points, plane: np.ndarray) -> str:
    """Fallback when an object covers the boundary and neither sky nor wall is adjacent."""
    array = np.asarray(points, dtype=float)
    dx = abs(float(array[-1][0] - array[0][0]))
    dy = abs(float(array[-1][1] - array[0][1]))
    rows = np.where(plane.any(axis=1))[0]
    if rows.size == 0:
        return "unknown"
    high = float(array[:, 1].mean()) <= (float(rows[0]) + float(rows[-1])) / 2.0
    slope = dy / max(dx, 1.0)
    if slope >= RIDGE_SLOPE_MAX:
        # A sloping edge high on the plane is a rake. Low on it, a gentle slope is an eave the
        # camera angle tilted; anything steeper is genuinely ambiguous.
        if high:
            return "rake"
        return "eave" if slope < np.tan(np.radians(EAVE_MAX_DEG)) else "unknown"
    return "ridge" if high else "eave"


def plane_side(mask: np.ndarray, image_width: int) -> str:
    """Which side of the house a whole roof plane faces.

    Decided once per plane and shared by every span on it. A span-by-span slope test calls a
    front gable's rakes "left" and "right" because they slope, so "front roofline" kept the
    one level eave and dropped every sloped edge of the very plane the customer asked for.
    """
    profile = bottom_profile(mask)
    if len(profile) < 2:
        return "front"
    ends = max(1, len(profile) // 20)
    chord = np.stack([profile[:ends].mean(axis=0), profile[-ends:].mean(axis=0)])
    if _classify(chord, image_width) == "front":
        return "front"
    eaves = [np.asarray(points) for points, _ in extract_spans(mask, "eave")]
    if eaves:
        longest = max(eaves, key=_length)
        if (_classify(longest, image_width) == "front"
                and _length(longest) >= FRONT_EAVE_SHARE * max(_plane_width(mask), 1)):
            return "front"
    return _classify(chord, image_width)


def _label_spans(spans: list[tuple], plane: np.ndarray, context, image_width: int) -> list[tuple]:
    """Side from the whole plane; edge by adjacency -> slope -> unknown."""
    side = plane_side(plane, image_width)
    return [
        (points, side, _adjacent_edge(points, plane, context, image_width) or _slope_edge(points, plane))
        for points, _, _ in spans
    ]


def seam_ridges(masks: list[np.ndarray], image_width: int) -> list[tuple]:
    """True ridges and hips: the shared border between two abutting plane masks.

    Only abutting pairs qualify. A mask whose own silhouette contains the apex holds its
    ridge in its interior, and no boundary segment of it is that ridge — those are left out
    rather than manufactured. A seam is a hip when it slopes like a roof corner and a ridge
    when level (see HIP_MIN_DEG); its side comes from the planes it separates.
    """
    if len(masks) < 2:
        return []
    size = max(3, int(SEAM_DILATE_FRACTION * image_width)) | 1
    kernel = np.ones((size, size), np.uint8)
    grown = [cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) for mask in masks]
    sides: list[str | None] = [None] * len(masks)

    found: list[tuple] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = float((masks[i] | masks[j]).sum())
            if union and float((masks[i] & masks[j]).sum()) / union > SEAM_MAX_IOU:
                continue
            overlap = grown[i] & grown[j]
            if overlap.sum() < 8:
                continue
            ys, xs = np.nonzero(overlap)
            cloud = np.stack([xs, ys], axis=1).astype(float)
            centre = cloud.mean(axis=0)
            axis = np.linalg.svd(cloud - centre, full_matrices=False)[2][0]
            along = (cloud - centre) @ axis
            start = centre + axis * float(np.percentile(along, 5))
            end = centre + axis * float(np.percentile(along, 95))
            if float(np.linalg.norm(end - start)) < SEAM_MIN_LEN * image_width:
                continue
            span = np.stack([start, end])
            slope = float(np.degrees(np.arctan2(abs(end[1] - start[1]), max(abs(end[0] - start[0]), 1e-6))))
            edge = "hip" if HIP_MIN_DEG <= slope <= HIP_MAX_DEG else "ridge"
            for k in (i, j):
                if sides[k] is None:
                    sides[k] = plane_side(masks[k], image_width)
            # A seam bounds both planes. Front wins: in a front-view photo its hips and ridges
            # belong to the face the customer is looking at.
            if "front" in (sides[i], sides[j]):
                side = "front"
            elif sides[i] == sides[j]:
                side = sides[i]
            else:
                side = _classify(span, image_width)
            found.append(
                ([[int(start[0]), int(start[1])], [int(end[0]), int(end[1])]], side, edge)
            )
    return found


def plane_spans(
    mask: np.ndarray, image_width: int, hybrid: bool = True, context=None
) -> list[tuple]:
    """The lines for one roof plane.

    Reading only the top and bottom profile is right when a plane spans the photo, but on
    a narrow plane it misses the gable rakes down the sides. Where the longest profile span
    fails to reach across the plane's own width, the outline polygon is used instead.
    Measuring reach against the plane rather than the image is what keeps this from
    disturbing the houses the profiles already handle well.
    """
    profile = [
        (points, side, edge)
        for edge in ("eave", "ridge")
        for points, side in extract_spans(mask, edge)
    ]
    if not hybrid:
        return _label_spans(profile, mask, context, image_width)
    if not profile:
        return _label_spans(_polygon_spans(mask, image_width), mask, context, image_width)
    longest = max(_length(np.asarray(points)) for points, *_ in profile)
    if longest >= PROFILE_MIN_FRAC * max(_plane_width(mask), 1):
        return _label_spans(profile, mask, context, image_width)
    return _label_spans(_polygon_spans(mask, image_width), mask, context, image_width)


def same_plane(index: int, masks: list[np.ndarray]) -> set[int]:
    """The masks that are this plane: itself, its duplicates, and masks one holds inside the other."""
    base = masks[index]
    area = float(base.sum())
    group = {index}
    for other_index, other in enumerate(masks):
        if other_index == index:
            continue
        shared = float((base & other).sum())
        if not shared:
            continue
        if (shared / float((base | other).sum()) > SEAM_MAX_IOU
                or shared / max(min(area, float(other.sum())), 1.0) > SAME_PLANE_CONTAINMENT):
            group.add(other_index)
    return group


def junction_share(points, index: int, masks: list[np.ndarray], image_width: int) -> float:
    """Share of a line's samples that sit on the plane's top edge with wall, then roof, above.

    Looks straight up from the top of the plane (and its duplicates) in each sample's column.
    A sample counts when another roof starts at least JUNCTION_GAP_FRACTION above that top and
    within JUNCTION_REACH of it. Samples off the plane's top edge never count, so an eave with
    the upper storey somewhere above it is not a junction.
    """
    group = same_plane(index, masks)
    rest = [mask for k, mask in enumerate(masks) if k not in group]
    if not rest:
        return 0.0
    plane = np.logical_or.reduce([masks[k] for k in group])
    above = np.logical_or.reduce(rest)
    height, width = plane.shape
    tops = plane.argmax(axis=0)
    filled = plane.any(axis=0)
    snap = JUNCTION_SNAP_FRACTION * image_width
    gap = JUNCTION_GAP_FRACTION * image_width
    reach = int(JUNCTION_REACH * height)

    samples = np.rint(_along(points)).astype(int)
    hits = 0
    for x, y in samples:
        if not (0 <= x < width and 0 <= y < height) or not filled[x] or abs(int(tops[x]) - y) > snap:
            continue
        top = int(tops[x])
        column = above[max(0, top - reach):top, x][::-1]
        if column.any() and int(column.argmax()) >= gap:
            hits += 1
    return hits / len(samples)


def mark_junctions(planes: list[list[tuple]], masks: list[np.ndarray], image_width: int) -> list[list[tuple]]:
    """Relabel each plane's top-edge lines that run into a wall with roof above as "junction".

    `planes[i]` holds the spans of `masks[i]`. Eaves are never tested: a junction is a top edge.
    """
    marked = []
    for index, spans in enumerate(planes):
        marked.append([
            (points, side, "junction")
            if edge != "eave" and junction_share(points, index, masks, image_width) >= JUNCTION_MIN_SHARE
            else (points, side, edge)
            for points, side, edge in spans
        ])
    return marked


def _chain(first: list, second: list) -> list | None:
    """One polyline from two that meet end to end, or None if no end meets another."""
    if first[-1] == second[0]:
        return first + second[1:]
    if first[-1] == second[-1]:
        return first + second[::-1][1:]
    if first[0] == second[-1]:
        return second + first[1:]
    if first[0] == second[0]:
        return second[::-1] + first[1:]
    return None


def _joinable(members: list[tuple]):
    """The first two spans with the same side and edge that meet end to end, and their join."""
    for i, first in enumerate(members):
        for second in members[i + 1:]:
            if first[1:] != second[1:]:
                continue
            # A closed outline has no free end left to extend.
            if first[0][0] == first[0][-1] or second[0][0] == second[0][-1]:
                continue
            points = _chain(first[0], second[0])
            if points is not None:
                return first, second, points
    return None


def join_pieces(spans: list[tuple], planes: list[list[tuple]]) -> list[tuple]:
    """Join pieces of one plane's edge that meet end to end into a single line.

    Extraction splits an edge wherever it turns, so one hip or one eave arrives as several
    panel entries. Only pieces of the same plane with the same side and edge type are joined,
    and only where they share an exact endpoint; lines from no plane (seams) pass through.
    `spans` must hold the very span objects in `planes` — what dedupe returns.
    """
    result = list(spans)
    for plane in planes:
        members = [span for span in result if any(span is own for own in plane)]
        while (pair := _joinable(members)) is not None:
            first, second, points = pair
            if points[0][0] > points[-1][0]:
                points = points[::-1]
            joined = (points, first[1], first[2])
            result = [joined if span is first else span for span in result if span is not second]
            members = [joined if span is first else span for span in members if span is not second]
    return result


def _resample(points: list[list[int]], count: int = 12) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    walked = np.concatenate([[0.0], np.cumsum(steps)])
    if walked[-1] <= 0:
        return np.repeat(array[:1], count, axis=0)
    targets = np.linspace(0.0, walked[-1], count)
    return np.stack(
        [np.interp(targets, walked, array[:, 0]), np.interp(targets, walked, array[:, 1])],
        axis=1,
    )


def dedupe(spans: list[tuple], width: int) -> list[tuple]:
    tolerance = max(4.0, 0.02 * width)
    kept: list[tuple] = []
    samples: list[np.ndarray] = []
    ordered = sorted(
        spans,
        key=lambda item: (EDGE_ORDER.get(item[2], len(EDGE_ORDER)), -_length(np.asarray(item[0]))),
    )
    for span in ordered:
        sample = _resample(span[0])
        if any(np.mean(np.linalg.norm(sample - other, axis=1)) < tolerance for other in samples):
            continue
        kept.append(span)
        samples.append(sample)
    return kept
