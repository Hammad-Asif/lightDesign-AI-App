"""The one roof-line representation every source is normalised into.

V3 answers in the pixel frame of the target-house crop it was shown; the partition geometry
(the fallback source) answers in photo pixels as (points, side, edge) spans. Either becomes
RoofLine objects in photo pixels before anything else touches them, so geometry, occlusion and
confidence are written once, against one type, and never see crop coordinates.

Coordinate spaces are explicit: `coordinate_space` is always "image" once a line exists here;
`source_coordinate_space` says what the source answered in ("crop" for V3), and `raw` keeps
the source's own answer untouched, so a V3 line can always be compared with what the model said.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np

Point = tuple[float, float]

# The edge types V3 returns and the editor knows. A type outside these is not invented or
# guessed at: the line is kept as "unknown", the same as the model's own "type uncertain".
EDGE_TYPES = ("eave", "rake", "hip", "ridge", "valley", "junction", "unknown")


# -- crop geometry --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CropBox:
    """A crop of the photo, in photo pixels: x0/y0 inclusive, x1/y1 exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]


def crop_to_image(point, crop: CropBox) -> Point:
    """A point in the crop's pixel frame, in photo pixels."""
    return float(point[0]) + crop.x0, float(point[1]) + crop.y0


def image_to_crop(point, crop: CropBox) -> Point:
    """A point in photo pixels, in the crop's pixel frame."""
    return float(point[0]) - crop.x0, float(point[1]) - crop.y0


MIN_CROP_PX = 32


def padded_box(x0: float, y0: float, x1: float, y1: float, padding: float, size: tuple[int, int]) -> CropBox:
    """A box grown by `padding` of its own width and height on each side, clamped to the photo."""
    width, height = size
    pad_x, pad_y = padding * max(x1 - x0, 1.0), padding * max(y1 - y0, 1.0)
    left, top = max(0, int(np.floor(x0 - pad_x))), max(0, int(np.floor(y0 - pad_y)))
    right, bottom = min(width, int(np.ceil(x1 + pad_x))), min(height, int(np.ceil(y1 + pad_y)))
    # Never a sliver: a model shown a 5 px strip has nothing to read.
    if right - left < MIN_CROP_PX:
        middle = (left + right) / 2
        left, right = max(0, int(middle - MIN_CROP_PX / 2)), min(width, int(middle + MIN_CROP_PX / 2))
    if bottom - top < MIN_CROP_PX:
        middle = (top + bottom) / 2
        top, bottom = max(0, int(middle - MIN_CROP_PX / 2)), min(height, int(middle + MIN_CROP_PX / 2))
    return CropBox(left, top, right, bottom)


# Sky-bounded top: a strip of sky this tall (share of the photo height, never under 12 px) is kept
# above where the sky begins, so the model sees the roof's top edge against it.
SKY_STRIP_FRACTION = 0.02
MIN_SKY_STRIP_PX = 12


def sky_start(union: np.ndarray, sky: np.ndarray, x0: int, x1: int) -> int | None:
    """The highest row where the sky begins above the roof, over the roof's columns, or None.

    Per column with roof: the lowest sky pixel above the roof's top; the sky "begins" just under
    it, and everything between there and the roof (an upper storey, a chimney, the top of a gable
    SAM missed) is not sky. A column with no sky above its roof (a tree, the photo's edge) has no
    say. The highest of the rest is taken, so the tallest part of the house is inside the crop.
    """
    columns = np.flatnonzero(union[:, x0:x1].any(axis=0)) + x0
    if not columns.size:
        return None
    tops = union[:, columns].argmax(axis=0)
    rows = np.arange(union.shape[0])[:, None]
    above = sky[:, columns] & (rows < tops[None, :])
    seen = above.any(axis=0)
    if not seen.any():
        return None
    lowest = union.shape[0] - 1 - above[::-1].argmax(axis=0)
    return int((lowest[seen] + 1).min())


def target_crop(masks: list[np.ndarray], keep: list[bool] | None, size: tuple[int, int],
                padding: float, sky: np.ndarray | None = None,
                sky_max_rise: float = 0.5) -> tuple[CropBox | None, str]:
    """The crop V3 is shown, and how its top was decided ("padding" or "sky"); None without planes.

    The padded bounding box of the target house's roof planes: sides and bottom by `padding`.
    With a sky mask, the top goes up to where the sky begins above the roof, plus a strip of sky,
    but never lower than the padded top and never more than `sky_max_rise` of the box's width above
    the roof, so a tree reaching into the sky cannot make the crop huge. No sky found: padding.

    `masks` and `sky` are photo-sized; `keep` is target.select's answer (None: every plane counts).
    """
    chosen = [m for m, k in zip(masks, keep if keep is not None else [True] * len(masks)) if k and m.any()]
    if not chosen:
        return None, "padding"
    union = np.logical_or.reduce(chosen)
    rows = np.flatnonzero(union.any(axis=1))
    cols = np.flatnonzero(union.any(axis=0))
    x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
    crop = padded_box(x0, y0, x1, y1, padding, size)
    if sky is None or not sky.any():
        return crop, "padding"
    start = sky_start(union, sky.astype(bool), x0, x1)
    if start is None:
        return crop, "padding"
    strip = max(MIN_SKY_STRIP_PX, int(round(SKY_STRIP_FRACTION * size[1])))
    highest = max(0, int(np.floor(y0 - sky_max_rise * (x1 - x0))))
    top = max(highest, start - strip)
    if top >= crop.y0:
        return crop, "padding"
    return CropBox(crop.x0, top, crop.x1, crop.y1), "sky"


# -- the line --------------------------------------------------------------------------------------


@dataclass
class RoofLine:
    """One roof line, from any source, in photo pixels.

    `points` is the polyline (V3 and partition lines have two points; hybrid extraction can
    give more). `source` names the source the line came from.
    Geometry stages set the metadata fields and may move `points` only where they say so
    (`snapped`, `occluded`); nothing else here changes a line's position.
    """

    points: list[list[float]]
    source: list[str]
    line_type: str | None
    confidence: float | None = None
    facade: str | None = None
    side: str | None = None
    vanishing_point: Point | None = None
    snapped: bool = False
    occluded: bool = False
    source_id: str | None = None
    coordinate_space: str = "image"
    source_coordinate_space: str = "image"
    # The source's own answer, untouched: V3's crop coordinates and type, or partition's span.
    raw: dict | None = None
    # Partition's connector across an occluded stretch, drawn from no pixels.
    inferred: bool = False
    snap_distance: float = 0.0
    snap_quality: float | None = None
    # Angle between the line and the direction to its vanishing point, when it has one.
    vp_angle: float | None = None
    # How the facade was decided: "vp" (points at one), "adjacent" (meets lines that do), or None.
    facade_basis: str | None = None
    # Why the plane filter kept or rejected it, and how far it sits from the house's planes.
    plane_status: str | None = None
    # source_ids of pieces joined into this line across a blocker.
    bridged_from: list[str] = field(default_factory=list)
    # The measured inputs of the confidence, kept so the number can be explained.
    evidence: dict = field(default_factory=dict)

    @property
    def start(self) -> Point:
        return tuple(self.points[0])  # type: ignore[return-value]

    @property
    def end(self) -> Point:
        return tuple(self.points[-1])  # type: ignore[return-value]

    @property
    def length(self) -> float:
        array = np.asarray(self.points, float)
        return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum()) if len(array) > 1 else 0.0

    @property
    def direction(self) -> np.ndarray:
        vector = np.asarray(self.points[-1], float) - np.asarray(self.points[0], float)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def int_points(self) -> list[list[int]]:
        return [[int(round(x)), int(round(y))] for x, y in self.points]

    def span(self) -> tuple:
        """The (points, side, edge) tuple the rest of the app has always used."""
        return self.int_points(), self.side, self.line_type or "unknown"

    def clone(self) -> "RoofLine":
        return copy.deepcopy(self)

    def summary(self) -> dict:
        """A JSON-safe record of the line for debugging: every field, rounded."""
        data = asdict(self)
        data["points"] = [[round(float(x), 1), round(float(y), 1)] for x, y in self.points]
        if self.vanishing_point is not None:
            data["vanishing_point"] = [round(float(v), 1) for v in self.vanishing_point]
        for key in ("snap_distance", "snap_quality", "vp_angle", "confidence"):
            if data.get(key) is not None:
                data[key] = round(float(data[key]), 3)
        return data


def _rightward(points: list[list[float]]) -> list[list[float]]:
    """Left end first, as partition orders its spans; the occlusion joins rely on it."""
    if len(points) >= 2 and (points[0][0], points[0][1]) > (points[-1][0], points[-1][1]):
        return points[::-1]
    return points


def grow_to(crop: CropBox, minimum: int, size: tuple[int, int]) -> CropBox:
    """The crop widened and heightened about its middle to at least `minimum` px, inside the photo."""
    width, height = size

    def span(start: int, end: int, limit: int) -> tuple[int, int]:
        if end - start >= minimum:
            return start, end
        if limit <= minimum:
            return 0, limit
        middle = (start + end) / 2
        start = int(round(middle - minimum / 2))
        start = min(max(0, start), limit - minimum)
        return start, start + minimum

    x0, x1 = span(crop.x0, crop.x1, width)
    y0, y1 = span(crop.y0, crop.y1, height)
    return CropBox(x0, y0, x1, y1)


def overlap_share(inner: CropBox, outer: CropBox) -> float:
    """How much of `inner`'s area lies inside `outer`."""
    w = max(0, min(inner.x1, outer.x1) - max(inner.x0, outer.x0))
    h = max(0, min(inner.y1, outer.y1) - max(inner.y0, outer.y0))
    area = inner.width * inner.height
    return w * h / area if area else 0.0


def normalize_v3_line(raw: dict, index: int, crop: CropBox, size: tuple[int, int], prefix: str = "v3") -> RoofLine | None:
    """One V3 answer line, from crop pixels into a RoofLine in photo pixels, or None if unusable.

    Nothing about the geometry is changed beyond the translation back to the photo (and a clamp
    inside it); the type is kept as the model gave it. No confidence is assigned here.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in raw["points"])
    except (KeyError, TypeError, ValueError):
        return None
    line_type = raw.get("type") if raw.get("type") in EDGE_TYPES else "unknown"
    width, height = size
    points = []
    for point in ((x1, y1), (x2, y2)):
        x, y = crop_to_image(point, crop)
        points.append([min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)])
    if np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]) < 1.0:
        return None
    return RoofLine(
        points=_rightward(points),
        source=["v3"],
        line_type=line_type,
        source_id=f"{prefix}_{index:03d}",
        source_coordinate_space="crop",
        raw={"type": raw.get("type"), "points": [x1, y1, x2, y2], "crop": crop.as_list()},
    )


def normalize_partition_line(span: tuple, index: int, inferred: set | None = None) -> RoofLine:
    """One partition (points, side, edge) span as a RoofLine, geometry and labels unchanged."""
    points, side, edge = span
    key = tuple(map(tuple, points))
    return RoofLine(
        points=[[float(x), float(y)] for x, y in points],
        source=["partition"],
        line_type=edge,
        side=side,
        source_id=f"partition_{index:03d}",
        raw={"points": [list(map(int, p)) for p in points], "side": side, "edge": edge},
        inferred=bool(inferred and key in inferred),
    )
