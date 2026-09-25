"""SAM 3 client (fal.ai, fal-ai/sam-3/image) and response decoding.

Measured behaviour, treated as fixed: a text prompt returns one instance per roof plane, with scores
around 0.5 to 0.7, and instances may overlap; point prompts are ignored unless the text prompt is
the empty string. Results are cached on disk by image content and prompt, so re-running a photo
costs nothing.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

SAM_API_URL = os.environ.get("SAM_API_URL", "https://fal.run/fal-ai/sam-3/image").strip()
SAM_API_KEY = (os.environ.get("FAL_KEY") or os.environ.get("SAM_API_KEY") or "").strip()
SAM_API_AUTH = os.environ.get("SAM_API_AUTH", "Key" if "fal.run" in SAM_API_URL else "Bearer").strip()
SAM_API_TIMEOUT = float(os.environ.get("SAM_API_TIMEOUT", "90"))
SAM_TEXT_PROMPT = os.environ.get("SAM_TEXT_PROMPT", "roof").strip() or "roof"
SAM_MAX_MASKS = int(os.environ.get("SAM_MAX_MASKS", "12"))
CACHE_DIR = BASE_DIR / ".sam_cache"


class SamError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def configured() -> bool:
    return bool(SAM_API_URL) and (bool(SAM_API_KEY) or "fal.run" not in SAM_API_URL)


def describe() -> dict:
    parts = urllib.parse.urlsplit(SAM_API_URL) if SAM_API_URL else None
    return {"sam_api": configured(), "sam_api_host": parts.netloc if parts else None, "sam_model": parts.path.strip("/") if parts else None,
            "sam_key_missing": bool(SAM_API_URL) and not SAM_API_KEY, "text_prompt": SAM_TEXT_PROMPT}


def build_sam_request(image_bgr: np.ndarray, text_prompt: str, point: Optional[tuple[int, int]] = None,
                      box: Optional[tuple[int, int, int, int]] = None) -> dict:
    """Request body in fal.ai's fal-ai/sam-3/image schema.

    The image travels inline as a data URI (fal accepts those in `image_url`), the masks come back
    as URLs that are then downloaded in parallel, and `apply_mask=False` asks for plain binary masks
    rather than a cut-out preview. Point prompts are in the pixel space of the image we send.

    Observed on fal: with a text prompt every instance of the concept comes back as its own mask
    (a house roof arrives as 5-10 sections); a point prompt is honoured only when `prompt` is
    the empty string, and then returns the object under the point.
    """
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise SamError(500, "Could not encode image for the SAM API.")
    body: dict = {
        "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
        "prompt": text_prompt,          # "" for a point-only call: fal ignores points when text is given
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": SAM_MAX_MASKS,
        "include_scores": True,
        "include_boxes": True,
        "sync_mode": False,        # masks come back as URLs: inline data made responses large enough to get cut off in transit
        "output_format": "png",
    }
    if point is not None:
        body["point_prompts"] = [{"x": int(point[0]), "y": int(point[1]), "label": 1}]
    if box is not None:
        body["box_prompts"] = [{"x_min": int(box[0]), "y_min": int(box[1]), "x_max": int(box[2]), "y_max": int(box[3])}]
    return body


def http_json(method: str, url: str, body: Optional[dict] = None) -> object:
    headers = {"Accept": "application/json"}
    if SAM_API_KEY:
        headers["Authorization"] = f"{SAM_API_AUTH} {SAM_API_KEY}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    last: Exception = RuntimeError("no attempt made")
    for attempt in range(2):                     # one retry: responses carry inline masks and occasionally get cut off
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=SAM_API_TIMEOUT) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")[:300]
            raise SamError(502, f"SAM API returned HTTP {err.code}: {detail}")
        except urllib.error.URLError as err:
            last = err
        except (http.client.HTTPException, ConnectionError, TimeoutError, json.JSONDecodeError) as err:
            last = err
        time.sleep(0.8)
    raise SamError(502, f"SAM API did not answer cleanly after a retry: {type(last).__name__}: {last}")


def call_sam_api(body: dict) -> object:
    """POST to the endpoint. fal's queue endpoint answers with status/response URLs to poll."""
    payload = http_json("POST", SAM_API_URL, body)
    if isinstance(payload, dict) and payload.get("status_url") and payload.get("response_url"):
        deadline = time.time() + SAM_API_TIMEOUT
        while time.time() < deadline:
            status = http_json("GET", payload["status_url"])
            state = status.get("status") if isinstance(status, dict) else None
            if state == "COMPLETED":
                return http_json("GET", payload["response_url"])
            if state in ("FAILED", "ERROR", "CANCELLED"):
                raise SamError(502, f"SAM API job {state.lower()}: {str(status)[:300]}")
            time.sleep(0.75)
        raise SamError(504, "SAM API job did not finish in time.")
    return payload


def decode_rle(rle: dict, fallback_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """COCO run-length encoding, both uncompressed (list counts) and compressed (string counts)."""
    size = rle.get("size") or fallback_shape
    h, w = int(size[0]), int(size[1])
    counts = rle.get("counts")
    if isinstance(counts, str):
        # LEB128-style variable-length integers with delta coding, as written by pycocotools.
        values, i, previous = [], 0, 0
        while i < len(counts):
            value, k, more = 0, 0, True
            while more:
                char = ord(counts[i]) - 48
                value |= (char & 0x1F) << (5 * k)
                more = bool(char & 0x20)
                i += 1
                k += 1
                if not more and (char & 0x10):
                    value |= -1 << (5 * k)
            if len(values) > 2:
                value += values[-2]
            values.append(value)
        counts = values
    if not isinstance(counts, list):
        return None
    flat = np.zeros(h * w, np.uint8)
    position, value = 0, 0
    for run in counts:
        run = int(run)
        if value:
            flat[position:position + run] = 255
        position += run
        value ^= 1
    return flat.reshape((w, h)).T  # COCO RLE is column-major


def decode_image_string(text: str) -> Optional[np.ndarray]:
    """A mask given as a data URL, a bare base64 PNG, or an http(s) URL to an image."""
    data: Optional[bytes] = None
    if text.startswith("data:image"):
        data = base64.b64decode(text.split(",", 1)[1])
    elif text.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(text, timeout=SAM_API_TIMEOUT) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError):
            return None
    elif len(text) > 64 and text[:8] in ("iVBORw0K", "/9j/4AAQ", "UklGRg=="):
        try:
            data = base64.b64decode(text)
        except ValueError:
            return None
    if data is None:
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[:, :, 3] if image.shape[2] == 4 and image[:, :, 3].min() == 0 else cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return np.where(image > 127, 255, 0).astype(np.uint8)


def polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def masks_from_response(payload: object, shape: tuple[int, int]) -> list[np.ndarray]:
    """Collect every mask-like thing in a JSON response, resized to the working image size.

    Recognised forms: data-URL / base64 / URL image strings, COCO RLE dicts, polygons
    ([[x, y], ...]), and 2-D 0/1 arrays. Keys are not assumed, so most providers work unchanged.
    """
    found: list[np.ndarray] = []

    def visit(node: object) -> None:
        if isinstance(node, str):
            mask = decode_image_string(node)
            if mask is not None:
                found.append(mask)
        elif isinstance(node, dict):
            if "counts" in node and "size" in node:
                mask = decode_rle(node, shape)
                if mask is not None:
                    found.append(mask)
                return
            for value in node.values():
                visit(value)
        elif isinstance(node, list) and node:
            first = node[0]
            if isinstance(first, (int, float)) and len(node) >= 6 and len(node) % 2 == 0 \
                    and all(isinstance(v, (int, float)) for v in node):
                found.append(polygon_mask(np.array(node, float).reshape(-1, 2), shape))   # COCO flat polygon
                return
            if isinstance(first, list) and first and isinstance(first[0], (int, float)):
                if len(node) == shape[0] and all(len(row) == shape[1] for row in node[:2]):
                    found.append(np.where(np.array(node) > 0, 255, 0).astype(np.uint8))     # 2-D array
                    return
                if len(node) >= 3 and all(len(row) == 2 for row in node):
                    found.append(polygon_mask(np.array(node, float), shape))                 # [[x, y], ...]
                    return
            for value in node:
                visit(value)

    visit(payload)
    resized = []
    for mask in found:
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.any():
            resized.append(mask)
    return resized


def masks_from_fal(payload: object, shape: tuple[int, int]) -> list[np.ndarray]:
    """fal-ai/sam-3/image output: `masks` is a list of image objects ({url, width, height, ...});
    `image` is the primary mask when multiple masks were not requested."""
    if not isinstance(payload, dict):
        return []
    items = payload.get("masks") or ([payload["image"]] if payload.get("image") else [])
    urls = [(item.get("url") if isinstance(item, dict) else item) for item in items]
    with ThreadPoolExecutor(max_workers=8) as pool:             # each mask is its own download
        decoded = list(pool.map(lambda u: decode_image_string(u) if isinstance(u, str) else None, urls))
    masks = []
    for mask in decoded:
        if mask is None:
            continue
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.any():
            masks.append(mask)
    return masks


def sam3_scores(payload: object, count: int) -> list[Optional[float]]:
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if isinstance(scores, list) and len(scores) == count:
        return [float(s) for s in scores]
    return [None] * count


# --------------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------------

def _cache_path(image_bgr: np.ndarray, key: str) -> Path:
    digest = hashlib.sha1(image_bgr.tobytes()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.npz"


def _cached(image_bgr: np.ndarray, key: str, produce) -> list[dict]:
    path = _cache_path(image_bgr, key)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return [{"mask": m, "score": None if np.isnan(s) else float(s), "source": key} for m, s in zip(data["masks"], data["scores"])]
    instances = produce()
    CACHE_DIR.mkdir(exist_ok=True)
    shape = image_bgr.shape[:2]
    np.savez_compressed(path, masks=np.stack([i["mask"] for i in instances]) if instances else np.zeros((0, *shape), np.uint8),
                        scores=np.array([np.nan if i["score"] is None else i["score"] for i in instances], np.float32))
    return instances


def _instances_from(payload: object, shape: tuple[int, int], source: str) -> list[dict]:
    masks = masks_from_fal(payload, shape) or masks_from_response(payload, shape)
    return [{"mask": m, "score": s, "source": source} for m, s in zip(masks, sam3_scores(payload, len(masks))) if m.any()]


def text_instances(image_bgr: np.ndarray, prompt: str) -> list[dict]:
    """Every instance of a concept: for "roof", one instance per roof plane (they may overlap)."""
    key = f"text:{prompt}:max{SAM_MAX_MASKS}"
    return _cached(image_bgr, key, lambda: _instances_from(call_sam_api(build_sam_request(image_bgr, prompt)), image_bgr.shape[:2], key))


def point_instance(image_bgr: np.ndarray, point: tuple[int, int]) -> list[dict]:
    """The object under one positive point. fal only honours points when the text prompt is empty."""
    key = f"point:{int(point[0])},{int(point[1])}"
    def produce():
        found = _instances_from(call_sam_api(build_sam_request(image_bgr, "", point)), image_bgr.shape[:2], key)
        x, y = int(point[0]), int(point[1])
        found.sort(key=lambda i: (bool(i["mask"][y, x]) if 0 <= y < i["mask"].shape[0] and 0 <= x < i["mask"].shape[1] else False, i["score"] or 0), reverse=True)
        return found[:1]
    return _cached(image_bgr, key, produce)


def box_instances(image_bgr: np.ndarray, prompt: str, box: tuple[int, int, int, int]) -> list[dict]:
    """A/B hook: does a box prompt work together with a non-empty text prompt? (Points do not.)"""
    key = f"box:{prompt}:{box}"
    return _cached(image_bgr, key, lambda: _instances_from(call_sam_api(build_sam_request(image_bgr, prompt, box=box)), image_bgr.shape[:2], key))


def concept_union(image_bgr: np.ndarray, prompts: list[str]) -> np.ndarray:
    """Union mask of other concepts ("tree", "dormer", "chimney", "skylight"), to subtract or distrust."""
    union = np.zeros(image_bgr.shape[:2], np.uint8)
    for prompt in prompts:
        for inst in text_instances(image_bgr, prompt):
            union |= inst["mask"]
    return union


def refine_small_planes(source_bgr: np.ndarray, working_bgr: np.ndarray, instances: list[dict], prompt: str,
                        max_area_frac: float = 0.01, pad: float = 0.35, target: int = 1024) -> list[dict]:
    """Optional second pass: re-prompt each small first-pass instance on a padded crop of the SOURCE
    resolution image, so small planes get more pixels, then map the sharper mask back."""
    H, W = working_bgr.shape[:2]
    sy, sx = source_bgr.shape[0] / H, source_bgr.shape[1] / W
    out = []
    for inst in instances:
        area = int((inst["mask"] > 0).sum())
        if area == 0 or area > max_area_frac * H * W:
            out.append(inst)
            continue
        ys, xs = np.nonzero(inst["mask"])
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        x0, x1 = max(0, int(xs.min() - pad * bw)), min(W, int(xs.max() + pad * bw) + 1)
        y0, y1 = max(0, int(ys.min() - pad * bh)), min(H, int(ys.max() + pad * bh) + 1)
        crop = source_bgr[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]
        scale = target / max(crop.shape[:2])
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        best, best_iou = None, 0.35
        reference = inst["mask"][y0:y1, x0:x1] > 0
        for cand in text_instances(crop, prompt):
            mask = cv2.resize(cand["mask"], (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
            iou = (mask & reference).sum() / max((mask | reference).sum(), 1)
            if iou > best_iou:
                best, best_iou = (mask, cand["score"]), iou
        if best is None:
            out.append(inst)
            continue
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1][best[0]] = 255
        out.append({"mask": full, "score": best[1] if best[1] is not None else inst["score"], "source": inst["source"] + "+crop"})
    return out
