from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass
class Detection:
    kind: str
    label: str
    mask: np.ndarray
    score: float
    box: tuple[int, int, int, int]


class SegmentationProvider(Protocol):
    def detect(self, image: Image.Image, kinds: list[str]) -> list[Detection]: ...

    def segment_box(
        self, image: Image.Image, box: tuple[int, int, int, int]
    ) -> np.ndarray: ...

    def segment_point(self, image: Image.Image, point: tuple[int, int]) -> np.ndarray: ...


def get_provider() -> SegmentationProvider:
    from app.config import get_settings

    settings = get_settings()
    choice = settings.segmentation_provider

    if choice == "fal":
        from app.providers.fal import FalProvider

        return FalProvider()

    # Never fall through to the mock silently: a stale "local" or "replicate" setting would
    # otherwise return placeholder shapes that look like real detections.
    if choice != "mock":
        raise RuntimeError(
            f"Unknown SEGMENTATION_PROVIDER {choice!r}. Use 'fal' or 'mock' — "
            "'local' and 'replicate' were removed along with the local models."
        )

    from app.providers.mock import MockProvider

    return MockProvider()
