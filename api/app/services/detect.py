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
    "palm_frond",
    "tree_canopy",
]

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


def expand_palms(detections: list[Detection]) -> list[Detection]:
    result: list[Detection] = []
    for item in detections:
        if item.kind != "palm_tree":
            result.append(item)
            continue
        parts = palm.split(item.mask, item.box)
        if parts is None:
            continue
        trunk, frond = parts
        if trunk.any():
            result.append(
                Detection(kind="palm_trunk", label="palm trunk", mask=trunk,
                          score=item.score, box=item.box)
            )
        if frond.any():
            result.append(
                Detection(kind="palm_frond", label="palm fronds", mask=frond,
                          score=item.score, box=item.box)
            )
    return result


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


def queried_kinds(selected: list[str]) -> list[str]:
    """What a request asks the model for: rooflines also look for things standing in front."""
    kinds = list(selected)
    if "roof" in kinds and "obstruction" not in kinds:
        kinds.append("obstruction")
    return kinds


def run(
    image: Image.Image,
    provider: SegmentationProvider,
    kinds: list[str] | None = None,
) -> list[Detection]:
    wanted = queried_kinds(kinds or DEFAULT_KINDS)

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
