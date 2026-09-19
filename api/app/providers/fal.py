"""SAM 3 via the fal.ai hosted API — the only detection path.

One endpoint serves all three prompt types: a text prompt returns every instance of a
concept, and box/point prompts drive the editor's re-segment and click-to-select tools.

Two invariants the rest of the pipeline silently depends on, enforced here in `_conform`:
masks come back as bool 2-D arrays in the *exact* frame of the image we posted, and the
box handed to `palm.split` lives in that same pixel space. `routers/regions.py` computes
roofline polylines straight from `mask.shape` without rescaling, so a frame mismatch
produces wrong geometry rather than an error.
"""

import base64
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import fal_client
import httpx
import numpy as np
from PIL import Image

from app.config import get_settings
from app.providers.base import Detection
from app.providers.prompts import (
    CLASS_QUERIES,
    NESTED,
    NESTED_SHARE,
    SCORE_THRESHOLD,
    acceptable,
    nms,
    suppress_across_classes,
)

log = logging.getLogger(__name__)

# Masks arrive antialiased on the PNG path; the repo's own convention is > 127.
MASK_THRESHOLD = 127
ASPECT_TOLERANCE = 0.02


def _data_uri(image: Image.Image, image_format: str = "png") -> str:
    buffer = io.BytesIO()
    if image_format == "jpeg":
        # The reference method posts JPEG at quality 92, encoded by OpenCV. Same encoder here, so
        # identical pixels give identical bytes and SAM 3 sees exactly what the reference sends.
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("could not encode the working copy as JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not rows.size or not cols.size:
        return (0, 0, 0, 0)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _conform(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Force a returned mask into the frame of the image we posted.

    fal's model page advertises a fixed 1024x1024 output while the SAM 3 model card says
    masks are resized to the input. Both cannot hold, so never trust the shape — check it.
    """
    width, height = size
    if mask.shape == (height, width):
        return mask

    posted = width / max(height, 1)
    returned = mask.shape[1] / max(mask.shape[0], 1)
    if abs(posted - returned) > ASPECT_TOLERANCE * posted:
        log.warning(
            "fal returned a mask at %sx%s for a %sx%s image — aspect ratio differs, so "
            "the resize below distorts it. Geometry from this mask is unreliable.",
            mask.shape[1], mask.shape[0], width, height,
        )
    else:
        log.info("resizing fal mask %sx%s to %sx%s", mask.shape[1], mask.shape[0], width, height)
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


MASK_FETCH_TIMEOUT = 15.0     # s per attempt: a stalled CDN connection must not cost the whole fal_timeout
MASK_FETCH_ATTEMPTS = 3


def _mask_from_url(url: str, size: tuple[int, int], timeout: float) -> np.ndarray | None:
    """Download and decode one mask. Short attempts with retries: a stalled connection used to wait
    the full API timeout (120 s) and then drop the plane, which showed up as minutes of detection
    time and missing roof lines."""
    last = None
    for attempt in range(MASK_FETCH_ATTEMPTS):
        try:
            data = httpx.get(url, timeout=min(timeout, MASK_FETCH_TIMEOUT)).content
            mask = np.array(Image.open(io.BytesIO(data)).convert("L")) > MASK_THRESHOLD
            return _conform(mask, size)
        except Exception as exc:  # noqa: BLE001 — one bad mask must not kill the batch
            last = exc
            log.warning("fetching a fal mask failed (attempt %d/%d): %s", attempt + 1, MASK_FETCH_ATTEMPTS, str(exc)[:140])
    log.warning("giving up on a fal mask after %d attempts: %s", MASK_FETCH_ATTEMPTS, str(last)[:140])
    return None


def _mask_from_rle(entry, size: tuple[int, int]) -> np.ndarray | None:
    """Decode one COCO-style RLE entry.

    fal documents this only as "Kaggle/COCO order" with no field-level schema, so this
    handles the uncompressed form (counts as a list of run lengths) and reports anything
    else loudly rather than guessing. Verify against the live API before relying on it.
    """
    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except ValueError:
            log.warning("fal RLE entry is a string but not JSON; cannot decode")
            return None
    if not isinstance(entry, dict) or "counts" not in entry or "size" not in entry:
        log.warning("unrecognised fal RLE shape: %s", type(entry).__name__)
        return None

    counts, (height, width) = entry["counts"], entry["size"]
    if not isinstance(counts, list):
        log.warning(
            "fal returned compressed RLE counts, which needs pycocotools. "
            "Set FAL_MASK_FORMAT=png until this is settled."
        )
        return None

    flat = np.zeros(height * width, dtype=bool)
    position, value = 0, False
    for run in counts:
        if value:
            flat[position : position + run] = True
        position += run
        value = not value
    # COCO RLE is column-major.
    return _conform(flat.reshape((width, height)).T, size)


def _drop_nested(
    found: dict[str, list[tuple[float, tuple]]],
    masks: dict[tuple[str, tuple], np.ndarray],
    shape: tuple[int, int],
) -> None:
    """Remove regions that sit inside a container region of another class (see NESTED)."""
    for outer, inner in NESTED:
        if not found.get(outer) or not found.get(inner):
            continue
        container = np.zeros(shape, dtype=bool)
        for _, box in found[outer]:
            container |= masks[(outer, box)]
        kept = []
        for score, box in found[inner]:
            mask = masks[(inner, box)]
            area = int(mask.sum())
            if area and int((mask & container).sum()) >= NESTED_SHARE * area:
                continue
            kept.append((score, box))
        found[inner] = kept


class FalProvider:
    def __init__(self, max_masks: int | None = None, image_format: str = "png"):
        self.settings = get_settings()
        if not self.settings.fal_key:
            raise RuntimeError("FAL_KEY is not set")
        # fal_client reads the key from the environment.
        os.environ.setdefault("FAL_KEY", self.settings.fal_key)
        # The request shape. Defaults are the settings, so the default path posts what it always
        # has; partition mode asks for the reference method's 12 instances and a JPEG.
        self.max_masks = max_masks if max_masks is not None else self.settings.fal_max_masks
        self.image_format = image_format

    # -- plumbing ---------------------------------------------------------------

    def _call(self, image: Image.Image, prompt: str, **extra) -> dict:
        arguments = {
            "image_url": _data_uri(image, self.image_format),
            "prompt": prompt,
            "apply_mask": False,
            "include_scores": True,
            "return_multiple_masks": True,
            "max_masks": self.max_masks,
            "output_format": "png",
            **extra,
        }
        try:
            return fal_client.subscribe(
                self.settings.fal_model,
                arguments=arguments,
                client_timeout=self.settings.fal_timeout,
            )
        except fal_client.FalClientTimeoutError as exc:
            raise RuntimeError(f"fal.ai timed out after {self.settings.fal_timeout}s") from exc
        except fal_client.FalClientError as exc:
            raise RuntimeError(f"fal.ai request failed: {str(exc)[:200]}") from exc

    def _instances(self, result: dict, size: tuple[int, int]) -> list[tuple[float, np.ndarray]]:
        """Every returned instance as (score, mask), scores matched by position.

        Scores are paired *before* anything is dropped. Filtering out an empty or failed
        mask first would slide every later mask onto its neighbour's score.
        """
        if self.settings.fal_mask_format == "rle":
            raw = result.get("rle") or []
            entries = [raw] if isinstance(raw, (str, dict)) else raw
            decoded = [_mask_from_rle(entry, size) for entry in entries]
        else:
            timeout = self.settings.fal_timeout
            items = result.get("masks") or []
            urls = [item["url"] if isinstance(item, dict) and item.get("url") else None for item in items]
            # Every mask is its own download (about 1.3 s each from fal's CDN): fetch them together.
            with ThreadPoolExecutor(max_workers=8) as pool:
                decoded = list(pool.map(lambda u: _mask_from_url(u, size, timeout) if u else None, urls))
        scores = result.get("scores") or []
        return [
            (float(scores[index]) if index < len(scores) else 0.5, mask)
            for index, mask in enumerate(decoded)
            if mask is not None and mask.any()
        ]

    # -- the Protocol -----------------------------------------------------------

    def detect(self, image: Image.Image, kinds: list[str]) -> list[Detection]:
        if not kinds:
            return []

        def one(kind: str):
            prompts = CLASS_QUERIES.get(kind, [])
            if not prompts:
                return kind, []
            return kind, self._instances(self._call(image, prompts[0]), image.size)

        workers = max(1, min(self.settings.fal_concurrency, len(kinds)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            gathered = list(pool.map(one, kinds))

        found: dict[str, list[tuple[float, tuple]]] = {}
        masks: dict[tuple[str, tuple], np.ndarray] = {}
        roofs: list[Detection] = []
        for kind, items in gathered:
            confident = [(score, mask) for score, mask in items if score >= SCORE_THRESHOLD]

            # Roof planes skip the box filters. Those limits were tuned on GroundingDINO
            # boxes, and on SAM 3's thin eave bands they discard real planes. Measured on
            # the 7 annotated houses: mean F1 66.0 without them, 64.8 with them. The local
            # SAM 3 roof path that set the 66.5 baseline never applied them either.
            if kind == "roof":
                roofs += [
                    Detection(kind="roof", label="roof", mask=mask, score=score, box=_bounds(mask))
                    for score, mask in confident
                ]
                continue

            # Everything else keeps the existing filters, each box tied to its own mask.
            boxes: list[tuple[float, tuple]] = []
            for score, mask in confident:
                box = _bounds(mask)
                if not acceptable(kind, box, image.size):
                    continue
                boxes.append((score, box))
                masks[(kind, box)] = mask
            if boxes:
                found[kind] = nms(boxes)

        suppress_across_classes(found)
        _drop_nested(found, masks, (image.height, image.width))

        return roofs + [
            Detection(
                kind=kind,
                label=kind.replace("_", " "),
                mask=masks[(kind, box)],
                score=score,
                box=box,
            )
            for kind, items in found.items()
            for score, box in items
            if (kind, box) in masks
        ]

    def segment_box(self, image: Image.Image, box) -> np.ndarray:
        x0, y0, x1, y1 = (int(v) for v in box)
        result = self._call(
            image,
            "",
            box_prompts=[{"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}],
            return_multiple_masks=False,
        )
        instances = self._instances(result, image.size)
        if instances:
            return max(instances, key=lambda item: item[0])[1]
        log.warning("fal returned no mask for a box prompt; falling back to the box itself")
        return _box_mask(image.size, (x0, y0, x1, y1))

    def segment_point(self, image: Image.Image, point) -> np.ndarray:
        x, y = (int(v) for v in point)
        result = self._call(
            image,
            "",
            point_prompts=[{"x": x, "y": y, "label": 1}],
            return_multiple_masks=False,
        )
        instances = self._instances(result, image.size)
        if instances:
            return max(instances, key=lambda item: item[0])[1]
        radius = max(image.width, image.height) // 20
        log.warning("fal returned no mask for a point prompt; falling back to a disc")
        return _box_mask(image.size, (x - radius, y - radius, x + radius, y + radius))


def _box_mask(size: tuple[int, int], box) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = (int(v) for v in box)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)] = True
    return mask
