import numpy as np
from PIL import Image

from app.providers.base import Detection

BOXES: dict[str, tuple[float, float, float, float]] = {
    "roof": (0.10, 0.18, 0.90, 0.42),
    "hedge": (0.06, 0.70, 0.42, 0.86),
    "shrub": (0.46, 0.74, 0.62, 0.88),
    "topiary": (0.66, 0.68, 0.78, 0.88),
    "palm_tree": (0.72, 0.08, 0.99, 0.92),
    "tree_canopy": (0.02, 0.30, 0.26, 0.62),
    "sky": (0.0, 0.0, 1.0, 0.10),
}


def _grid(width: int, height: int, box: tuple[float, float, float, float]):
    x0, y0, x1, y1 = box
    ys, xs = np.mgrid[0:height, 0:width]
    return xs / width, ys / height, x0, y0, x1, y1


def _gable(width: int, height: int, box) -> np.ndarray:
    xs, ys, x0, y0, x1, y1 = _grid(width, height, box)
    mid = (x0 + x1) / 2
    ridge = y0 + (y1 - y0) * 0.25
    slope = (y1 - ridge) / max(mid - x0, 1e-6)
    top = ridge + np.abs(xs - mid) * slope
    return (xs >= x0) & (xs <= x1) & (ys >= top) & (ys <= y1)


def _ellipse(width: int, height: int, box) -> np.ndarray:
    xs, ys, x0, y0, x1, y1 = _grid(width, height, box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = max((x1 - x0) / 2, 1e-6), max((y1 - y0) / 2, 1e-6)
    return ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0


def _rect(width: int, height: int, box) -> np.ndarray:
    xs, ys, x0, y0, x1, y1 = _grid(width, height, box)
    return (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)


def _palm(width: int, height: int, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    crown_bottom = y0 + (y1 - y0) * 0.32
    mid = (x0 + x1) / 2
    trunk_half = (x1 - x0) * 0.06
    crown = _ellipse(width, height, (x0, y0, x1, crown_bottom))
    trunk = _rect(width, height, (mid - trunk_half, crown_bottom, mid + trunk_half, y1))
    return crown | trunk


SHAPES = {
    "roof": _gable,
    "hedge": _rect,
    "shrub": _ellipse,
    "topiary": _ellipse,
    "palm_tree": _palm,
    "tree_canopy": _ellipse,
    "sky": _rect,
}


class MockProvider:
    def detect(self, image: Image.Image, kinds: list[str]) -> list[Detection]:
        width, height = image.size
        results: list[Detection] = []
        for kind in kinds:
            box = BOXES.get(kind)
            if box is None:
                continue
            mask = SHAPES[kind](width, height, box)
            if not mask.any():
                continue
            x0, y0, x1, y1 = box
            results.append(
                Detection(
                    kind=kind,
                    label=kind.replace("_", " "),
                    mask=mask,
                    score=0.5,
                    box=(
                        int(x0 * width),
                        int(y0 * height),
                        int(x1 * width),
                        int(y1 * height),
                    ),
                )
            )
        return results

    def segment_box(self, image: Image.Image, box) -> np.ndarray:
        width, height = image.size
        x0, y0, x1, y1 = box
        norm = (x0 / width, y0 / height, x1 / width, y1 / height)
        return _ellipse(width, height, norm)

    def segment_point(self, image: Image.Image, point) -> np.ndarray:
        width, height = image.size
        cx, cy = point[0] / width, point[1] / height
        radius = 0.08
        return _ellipse(width, height, (cx - radius, cy - radius, cx + radius, cy + radius))
