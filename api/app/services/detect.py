import cv2
import numpy as np
from PIL import Image

from app.providers.base import Detection, SegmentationProvider
from app.providers.prompts import query_kinds
from app.services import palm

DEFAULT_KINDS = [
    "roof",
    "hedge",
    "shrub",
    "topiary",
    "palm_trunk",
    "palm_collar",
    "palm_frond",
    "tree_canopy",
    # Only in the "detect everything" pass: an instruction that says roofline gets a roofline.
    "obstruction",
]

# One colour per roof edge role, the reference app's palette, so a glance tells a ridge from an
# eave. Edges the role step could not name keep the plain roof colour.
EDGE_COLORS = {"ridge": "#ff3b30", "hip": "#ff9f0a", "valley": "#0a84ff", "eave": "#39ff14", "rake": "#bf5af2"}


def edge_color(edge: str | None) -> str:
    return EDGE_COLORS.get(edge or "", COLORS["roof"])


COLORS = {
    "roof": "#38bdf8",
    "driveway_edge": "#a855f7",
    "walkway_edge": "#d946ef",
    "fence": "#6366f1",
    "column": "#f472b6",
    "window": "#fde047",
    "door": "#fb923c",
    "garage_door": "#ea580c",
    "wreath": "#ef4444",
    "hedge": "#22c55e",
    "shrub": "#84cc16",
    "topiary": "#14b8a6",
    "palm_trunk": "#f59e0b",
    "palm_collar": "#e11d48",
    "palm_frond": "#a3e635",
    "tree_canopy": "#059669",
    "obstruction": "#94a3b8",
    "custom": "#ec4899",
}

LABELS = {
    "roof": "Roofline",
    "driveway_edge": "Driveway edge",
    "walkway_edge": "Walkway edge",
    "fence": "Fence / railing",
    "column": "Column / post",
    "window": "Window",
    "door": "Entry door",
    "garage_door": "Garage door",
    "wreath": "Wreath",
    "hedge": "Hedge",
    "shrub": "Shrub",
    "topiary": "Topiary",
    "palm_trunk": "Palm trunk",
    "palm_collar": "Palm collar",
    "palm_frond": "Palm fronds",
    "tree_canopy": "Tree canopy",
    "obstruction": "Obstruction",
    "custom": "Custom region",
}

LINE_KINDS = {"roof", "driveway_edge", "walkway_edge", "fence"}

MIN_AREA_FRACTION = 0.0004


def clean(mask: np.ndarray, image_area: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    minimum = MIN_AREA_FRACTION * image_area
    keep = np.zeros_like(binary, dtype=bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            keep |= labels == index
    return keep if keep.any() else binary.astype(bool)


# Two palm instances are the same palm when their trunks overlap this much. Crowns are not
# compared: neighbouring palms interleave their fronds, and it is the trunk that stands in front
# of a roofline.
SAME_TRUNK_IOU = 0.5


def expand_palms(detections: list[Detection]) -> list[Detection]:
    # Split every palm first, then decide duplicates by trunk in score order: the same palm
    # twice is dropped, two palms standing together — one crown box, two trunks — both stay.
    split = {id(item): palm.split(item.mask, item.box) for item in detections if item.kind == "palm_tree"}
    kept: set[int] = set()
    trunks: list[np.ndarray] = []
    for item in sorted((d for d in detections if d.kind == "palm_tree"), key=lambda d: -d.score):
        parts = split[id(item)]
        if parts is None:
            continue
        trunk = parts[0]
        if trunk.any() and any(_iou(trunk, seen) >= SAME_TRUNK_IOU for seen in trunks):
            continue
        trunks.append(trunk)
        kept.add(id(item))

    result: list[Detection] = []
    for item in detections:
        if item.kind != "palm_tree":
            result.append(item)
            continue
        if id(item) not in kept:
            continue
        # Three zones lit independently: a warm white trunk, a red collar, green fronds.
        for kind, label, mask in zip(
            ("palm_trunk", "palm_collar", "palm_frond"), ("palm trunk", "palm collar", "palm fronds"), split[id(item)]
        ):
            if mask.any():
                result.append(Detection(kind=kind, label=label, mask=mask, score=item.score, box=item.box))
    return result


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return int(np.count_nonzero(a & b)) / union if union else 0.0


def working_copy(image: Image.Image, max_edge: int) -> tuple[Image.Image, float]:
    if max(image.size) <= max_edge:
        return image, 1.0
    scale = max_edge / max(image.size)
    size = (max(round(image.width * scale), 1), max(round(image.height * scale), 1))
    return image.resize(size, Image.LANCZOS), image.width / size[0]


def upscale_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.shape[1] == size[0] and mask.shape[0] == size[1]:
        return mask
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


# What can stand between the camera and a roofline. Detecting them lets a line that disappears
# behind a palm be joined up and its hidden stretch reported — but they are not what the customer
# asked for, so they are only fetched when blockers are switched on (Settings.detect_blockers).
# The whole palm counts, not just its trunk: on a two-storey house it is the fronds that hang
# in front of the upper roof, and all three parts come out of the one "palm tree" call.
ROOF_BLOCKERS = ["obstruction", "palm_trunk", "palm_collar", "palm_frond", "tree_canopy"]


def queried_kinds(selected: list[str], blockers: bool = False) -> list[str]:
    """What a request asks the model for. An instruction names what it wants and gets that."""
    kinds = list(selected)
    if blockers and "roof" in kinds:
        kinds += [kind for kind in ROOF_BLOCKERS if kind not in kinds]
    return kinds


def run(
    image: Image.Image,
    provider: SegmentationProvider,
    kinds: list[str] | None = None,
    blockers: bool = False,
) -> list[Detection]:
    wanted = queried_kinds(kinds or DEFAULT_KINDS, blockers)

    # Every kind, roof included, is one text prompt to the same model.
    detections = provider.detect(image, query_kinds(wanted))

    # Roof masks stay exempt from `clean()`, exactly as they were on the local SAM 3 path,
    # so this migration changes the model source and nothing else. Whether cleaning helps
    # the roofline is a separate measurement against tests/score.py.
    image_area = image.width * image.height
    for detection in detections:
        if detection.kind != "roof":
            detection.mask = clean(detection.mask, image_area)

    detections = expand_palms([d for d in detections if d.mask.any()])
    return [d for d in detections if d.kind in wanted and d.mask.any()]


def label_for(kind: str, index: int, total: int) -> str:
    base = LABELS.get(kind, kind.replace("_", " ").title())
    return f"{base} {index + 1}" if total > 1 else base
