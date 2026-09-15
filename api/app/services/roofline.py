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
    minimum = MIN_SPAN_FRACTION * width
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
    return [
        ([a.tolist(), b.tolist()], "front", "eave")
        for a, b in zip(corners, np.roll(corners, -1, axis=0))
        if float(np.linalg.norm(b.astype(float) - a)) >= POLY_MIN_LEN * image_width
    ]


def plane_spans(mask: np.ndarray, image_width: int, hybrid: bool = True) -> list[tuple]:
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
        return profile
    if not profile:
        return _polygon_spans(mask, image_width)
    longest = max(_length(np.asarray(points)) for points, *_ in profile)
    if longest >= PROFILE_MIN_FRAC * max(_plane_width(mask), 1):
        return profile
    return _polygon_spans(mask, image_width)


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
        key=lambda item: (item[2] != "eave", -_length(np.asarray(item[0]))),
    )
    for span in ordered:
        sample = _resample(span[0])
        if any(np.mean(np.linalg.norm(sample - other, axis=1)) < tolerance for other in samples):
            continue
        kept.append(span)
        samples.append(sample)
    return kept
