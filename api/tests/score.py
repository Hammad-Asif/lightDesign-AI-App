"""Standalone scorer: measures any set of roofline polylines against the annotations.

Independent of the detection pipeline, so it can compare different detector builds.

    from tests.score import load_targets, score, PAIRS
"""
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PHOTOS = Path("/Users/samibutt/Documents/freelance/Hammad-project/bulb-project/lighting project")
CLEAN_DIR = PHOTOS / "truth_clean"     # cleaned annotations win when present
MAX_EDGE, MIN_EDGE = 1280, 1000
TOLERANCE = 8.0
MIN_STROKE = 150
DIAGONAL_DEG = 20.0


def working(path: Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    longest = max(image.size)
    if longest > MAX_EDGE:
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
    elif longest < MIN_EDGE:
        f = MIN_EDGE / longest
        image = image.resize((round(image.width * f), round(image.height * f)), Image.LANCZOS)
    return image


def _centreline(mask: np.ndarray) -> np.ndarray:
    points = []
    for x in range(mask.shape[1]):
        rows = np.where(mask[:, x])[0]
        if rows.size == 0:
            continue
        for run in np.split(rows, np.where(np.diff(rows) > 1)[0] + 1):
            if run.size:
                points.append((x, float(run.mean())))
    return np.asarray(points, dtype=float) if points else np.zeros((0, 2))


def load_targets(annotation: Path, size: tuple[int, int]) -> np.ndarray:
    a = np.asarray(Image.open(annotation).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 120) & (r - g > 60) & (r - b > 60)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_STROKE:
            keep |= labels == i
    return _centreline(keep) * (size[0] / a.shape[1])


def distance(points: np.ndarray, spans: list) -> np.ndarray:
    if not spans or points.size == 0:
        return np.full(len(points), np.inf)
    best = np.full(len(points), np.inf)
    for span, *_ in spans:
        arr = np.asarray(span, dtype=float)
        for u, v in zip(arr, arr[1:]):
            d = v - u
            L = float(np.dot(d, d))
            if L < 1e-6:
                continue
            t = np.clip(((points - u) @ d) / L, 0.0, 1.0)
            best = np.minimum(best, np.linalg.norm(points - (u + t[:, None] * d), axis=1))
    return best


def line_points(spans: list, step: float = 2.0) -> np.ndarray:
    out = []
    for span, *_ in spans:
        arr = np.asarray(span, dtype=float)
        for u, v in zip(arr, arr[1:]):
            L = float(np.linalg.norm(v - u))
            if L < 1e-6:
                continue
            for t in np.linspace(0.0, 1.0, max(int(L / step), 2)):
                out.append(u + t * (v - u))
    return np.asarray(out) if out else np.zeros((0, 2))


def score(spans: list, targets: np.ndarray) -> tuple[float, float, float, int]:
    if not spans or targets.size == 0:
        return 0.0, 0.0, 0.0, 0
    recall = 100.0 * (distance(targets, spans) <= TOLERANCE).mean()
    drawn = line_points(spans)
    precision = (
        100.0 * (distance(drawn, [(targets.tolist(),)]) <= TOLERANCE).mean() if drawn.size else 0.0
    )
    return recall, precision, 2 * recall * precision / max(recall + precision, 1e-6), len(spans)


def pairs() -> list[tuple[str, Path, Path]]:
    found = []
    for annotation in sorted(PHOTOS.glob("ground_truth_*")):
        if annotation.is_dir():
            continue
        cleaned = CLEAN_DIR / annotation.name
        if cleaned.exists():
            annotation = cleaned
        key = re.sub(r"[^a-z0-9]", "", annotation.stem[len("ground_truth_"):].lower())
        for photo in sorted(PHOTOS.iterdir()):
            if photo.suffix.lower() not in {".jpg", ".jpeg", ".png"} or photo.name.startswith("ground_truth_"):
                continue
            if key and key in re.sub(r"[^a-z0-9]", "", photo.stem.lower()):
                found.append((photo.stem, photo, annotation))
                break
    return found
