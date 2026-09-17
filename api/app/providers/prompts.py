# SAM 3 takes a short noun phrase and returns every instance of that concept, so each
# class needs exactly one prompt and no trailing "." — the period was a GroundingDINO
# convention. These phrases are SAM 3-native and differ sharply from the DINO-tuned ones
# they replace: measured on the same photos, "bush" scored 5/6 where "round bush" scored
# 1/6, and "row of low bushes" scored 5/5 where both "hedge" and "trimmed hedge" scored 0/5.
CLASS_QUERIES: dict[str, list[str]] = {
    "roof": ["roof"],
    "hedge": ["row of low bushes"],
    "shrub": ["bush"],
    "topiary": ["topiary"],
    "palm_tree": ["palm tree"],
    "tree_canopy": ["tree"],
    # Asked alongside rooflines (detect.queried_kinds) for what stands in front of the house.
    # Palms, trees and planting block lines too, but already have classes of their own.
    "obstruction": ["car"],
}

# Matches the threshold the local SAM 3 roof path used.
SCORE_THRESHOLD = 0.4

MAX_PER_CLASS = 8
NMS_IOU = 0.6

BOX_LIMITS: dict[str, dict[str, float]] = {
    "roof": {"max_area": 0.55, "max_aspect": 0.85, "min_width": 0.12},
    "topiary": {"max_area": 0.06},
}

COMPETING = [
    ("palm_tree", "tree_canopy"),
    ("palm_tree", "topiary"),
    ("tree_canopy", "topiary"),
    ("hedge", "shrub"),
    ("shrub", "topiary"),
    ("hedge", "topiary"),
]

# (container, contained). SAM 3 returns a hedge row as a hedge and also each bush inside it as
# a shrub, so the same plants came back as two overlapping regions: 26 of 41 shrubs on the
# client's photos sat inside a detected hedge. Judged on mask pixels, not boxes, because a
# long hedge's box also spans lawn and beds the hedge itself doesn't cover.
NESTED = [("hedge", "shrub"), ("hedge", "topiary")]
NESTED_SHARE = 0.8

PALM_KINDS = {"palm_trunk", "palm_collar", "palm_frond"}


def query_kinds(kinds: list[str]) -> list[str]:
    resolved = [k for k in kinds if k in CLASS_QUERIES]
    if PALM_KINDS & set(kinds):
        resolved.append("palm_tree")
    return resolved


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def nms(boxes: list[tuple[float, tuple]], threshold: float = NMS_IOU) -> list[tuple[float, tuple]]:
    keep: list[tuple[float, tuple]] = []
    for score, box in sorted(boxes, key=lambda item: -item[0]):
        if all(iou(box, kept) < threshold for _, kept in keep):
            keep.append((score, box))
    return keep[:MAX_PER_CLASS]


def covered_by(inner: tuple, outer: tuple) -> float:
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ((x1 - x0) * (y1 - y0)) / area if area else 0.0


def acceptable(kind: str, box: tuple, size: tuple[int, int]) -> bool:
    width, height = box[2] - box[0], box[3] - box[1]
    if width < 6 or height < 6:
        return False
    limits = BOX_LIMITS.get(kind)
    if limits is None:
        return True
    image_width, image_height = size
    if "max_area" in limits and width * height > limits["max_area"] * image_width * image_height:
        return False
    if "max_aspect" in limits and height > limits["max_aspect"] * width:
        return False
    if "min_width" in limits and width < limits["min_width"] * image_width:
        return False
    return True


def collapse_facade_boxes(boxes: list[tuple[float, tuple]], image_height: int) -> list[tuple[float, tuple]]:
    top_tolerance = max(4.0, 0.02 * image_height)
    dropped: set[int] = set()
    for i, (_, tall) in enumerate(boxes):
        if i in dropped:
            continue
        for j, (_, tight) in enumerate(boxes):
            if i == j or j in dropped:
                continue
            tall_h, tight_h = tall[3] - tall[1], tight[3] - tight[1]
            if tall_h <= 1.5 * tight_h or abs(tall[1] - tight[1]) > top_tolerance:
                continue
            overlap = max(0, min(tall[2], tight[2]) - max(tall[0], tight[0]))
            spread = max(tall[2], tight[2]) - min(tall[0], tight[0])
            if spread and overlap / spread > 0.75:
                dropped.add(i)
                break
    return [box for k, box in enumerate(boxes) if k not in dropped]


def suppress_across_classes(found: dict[str, list[tuple[float, tuple]]]) -> None:
    for first, second in COMPETING:
        if first not in found or second not in found:
            continue
        losers = []
        for score, box in found[second]:
            rival = max(
                (s for s, b in found[first] if iou(box, b) > 0.55 or covered_by(box, b) > 0.6),
                default=None,
            )
            if rival is not None and rival >= score:
                losers.append((score, box))
        for item in losers:
            found[second].remove(item)
