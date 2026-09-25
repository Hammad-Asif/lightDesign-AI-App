"""
Roofline extractor -- backend.

A small FastAPI service that turns a photo of a house into a vector roofline.

Pipeline
--------
1. POST /api/upload    Store the image in an in-memory session (downscaled to a working size).
2. POST /api/segment   Produce one mask per roof section ("instance"). Three strategies:
                         "sam3"       SAM 3 on fal.ai (fal-ai/sam-3/image). Enabled when FAL_KEY is
                                      set. The text prompt ("roof") returns every roof section as
                                      its own instance; each point prompt adds one more section.
                         "mock"       Local stand-in with no API key: GrabCut seeded from each point.
                         "threshold"  Plain grayscale threshold driven by a slider (one instance).
3. POST /api/geometry  Turn the cached instances into straight roof vectors, per section:
                         cv2.findContours -> cv2.approxPolyDP(epsilon) for the corner topology
                         -> least-squares refit of every edge on the dense contour pixels
                         -> merge near-collinear neighbours -> vertices = line intersections
                         -> merge edges shared by two sections (hips, valleys)
                         -> height-cutoff filter -> slope / angle / pitch / role per segment.

The geometry step is cheap, so the frontend calls it on every slider tick.

Run:  uvicorn app:app --reload
      SAM 3 (fal.ai) switches on when FAL_KEY is set in .env (see .env.example).
"""

from __future__ import annotations

import base64
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

# .env next to app.py is loaded automatically (python-dotenv ships with uvicorn[standard]).
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Uploads are downscaled so the long edge is at most this many pixels. All coordinates in the
# API and the export refer to this working image; `scale_from_source` lets callers map back.
WORKING_MAX_DIM = 2000
# GrabCut is O(pixels * iterations); run it on a smaller copy and upscale the mask afterwards.
GRABCUT_MAX_DIM = 720
GRABCUT_ITERATIONS = 5
MAX_SESSIONS = 12
# Contours smaller than this fraction of the largest one are ignored (specks, distant roofs).
MIN_CONTOUR_FRACTION = 0.05
MAX_CONTOURS = 8

# SAM 3 via fal.ai: https://fal.ai/models/fal-ai/sam-3/image. Only FAL_KEY is required; without it
# the app runs fully offline with the mock. SAM_API_URL can point at the queue endpoint
# (https://queue.fal.run/fal-ai/sam-3/image) or at another provider with the same schema.
SAM_API_URL = os.environ.get("SAM_API_URL", "https://fal.run/fal-ai/sam-3/image").strip()
SAM_API_KEY = (os.environ.get("FAL_KEY") or os.environ.get("SAM_API_KEY") or "").strip()
SAM_API_AUTH = os.environ.get("SAM_API_AUTH", "Key" if "fal.run" in SAM_API_URL else "Bearer").strip()
SAM_API_TIMEOUT = float(os.environ.get("SAM_API_TIMEOUT", "90"))
SAM_TEXT_PROMPT = os.environ.get("SAM_TEXT_PROMPT", "roof").strip() or "roof"
SAM_MAX_MASKS = int(os.environ.get("SAM_MAX_MASKS", "12"))
MIN_INSTANCE_FRACTION = 0.0002   # instances smaller than this fraction of the image are noise

app = FastAPI(title="Roofline extractor")


# --------------------------------------------------------------------------------------------
# Session store (in-memory, oldest evicted first)
# --------------------------------------------------------------------------------------------

class Session:
    def __init__(self, image: np.ndarray, source_size: tuple[int, int], filename: str):
        self.image = image                      # BGR working image
        self.source_size = source_size          # (width, height) of the uploaded file
        self.filename = filename
        self.encoded = encode_jpeg(image)       # bytes served to the browser
        self.instances: list[dict] = []         # {"mask": uint8 0/255, "score", "source", "color"}
        self.contours: list[tuple[int, np.ndarray]] = []   # (instance index, dense external contour)
        self.segmentation: dict = {}            # parameters that produced the instances
        self.cache: dict[str, list[dict]] = {}  # prompt key -> instances, so a new click costs one call


SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


def get_session(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found. Upload the image again.")
    return session


def put_session(session: Session) -> str:
    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = session
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return session_id


# --------------------------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "That file is not an image OpenCV can read (use JPG, PNG or WebP).")
    return image


def fit_to_working_size(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = WORKING_MAX_DIM / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = 94) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "Could not encode image.")
    return buf.tobytes()


def encode_png_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(500, "Could not encode mask.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------------------------
# Mask post-processing shared by every segmentation strategy
# --------------------------------------------------------------------------------------------

def keep_component_at(mask: np.ndarray, point: Optional[tuple[int, int]]) -> np.ndarray:
    """Keep the connected component containing `point`, or the largest one if that misses."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    label = 0
    if point is not None:
        x, y = point
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            label = int(labels[y, x])
    if label == 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes (windows, texture) so each blob is a single clean silhouette."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def usable_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Dense external contours worth vectorising, largest first.

    CHAIN_APPROX_NONE keeps every boundary pixel: the least-squares refit needs them all.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if len(c) >= 3]
    if not contours:
        return []
    areas = [cv2.contourArea(c) for c in contours]
    largest = max(areas)
    keep = [c for c, a in zip(contours, areas) if a > 0 and a >= MIN_CONTOUR_FRACTION * largest]
    keep.sort(key=cv2.contourArea, reverse=True)
    return keep[:MAX_CONTOURS]


def clean_instance(mask: np.ndarray) -> Optional[np.ndarray]:
    """Tidy one section mask: close notches, drop specks, fill holes, keep blobs near the largest
    one's size. Roof sections are near-convex, so a modest closing (0.8 % of the image) only
    removes SAM's boundary wobble and bites taken out by tile texture."""
    k = max(3, int(0.008 * max(mask.shape)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if not areas or max(areas) <= 0:
        return None
    keep = [c for c, a in zip(contours, areas) if a >= MIN_CONTOUR_FRACTION * max(areas)]
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, keep, -1, 255, thickness=cv2.FILLED)
    for c in keep:
        fill_narrow_notches(filled, c, min_depth=0.01 * max(mask.shape))
    if np.count_nonzero(filled) < MIN_INSTANCE_FRACTION * mask.size:
        return None
    return filled


def fill_narrow_notches(mask: np.ndarray, contour: np.ndarray, min_depth: float,
                        max_mouth_ratio: float = 2.0, max_area_fraction: float = 0.08) -> None:
    """Fill bites taken out of a section's edge (tile texture, a chimney shadow) in place.

    A bite is a convexity defect that is deeper than `min_depth`, whose mouth is narrower than
    `max_mouth_ratio` times its depth, and which covers at most `max_area_fraction` of the
    section. Wide or large concavities, such as the inside corner of an L-shaped roof or the
    bay where a neighbouring section nests, are left alone.
    """
    section_area = cv2.contourArea(contour)
    if len(contour) < 4:
        return
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return
    if defects is None:
        return
    for start, end, far, depth in defects.reshape(-1, 4):
        depth_px = depth / 256.0
        if depth_px < min_depth:
            continue
        a, b = contour[start, 0], contour[end, 0]
        mouth = float(np.linalg.norm(a - b))
        if mouth > max_mouth_ratio * depth_px:
            continue
        # the notch is the boundary run from `start` to `end`, closed by the hull chord
        run = contour[start:end + 1, 0] if end >= start else np.concatenate([contour[start:, 0], contour[:end + 1, 0]])
        if len(run) < 3 or cv2.contourArea(run.astype(np.int32).reshape(-1, 1, 2)) > max_area_fraction * section_area:
            continue
        cv2.fillPoly(mask, [run.astype(np.int32).reshape(-1, 1, 2)], 255)


def instance_color(index: int) -> tuple[int, int, int]:
    """Distinct BGR colour per section (golden-angle hue spacing)."""
    hue = int((index * 137.508) % 360) // 2
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 190, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def make_instance(mask: np.ndarray, source: str, score: Optional[float] = None) -> Optional[dict]:
    cleaned = clean_instance(mask)
    if cleaned is None:
        return None
    return {"mask": cleaned, "source": source, "score": None if score is None else round(float(score), 4)}


def render_instances(instances: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Viewport A: a single section stays a plain binary mask; several get distinct colours."""
    if len(instances) == 1:
        return instances[0]["mask"]
    canvas = np.zeros((shape[0], shape[1], 3), np.uint8)
    for i, inst in enumerate(instances):
        canvas[inst["mask"] > 0] = instance_color(i)
    return canvas


# --------------------------------------------------------------------------------------------
# Segmentation: SAM 3 API
# --------------------------------------------------------------------------------------------

def sam_api_configured() -> bool:
    return bool(SAM_API_URL) and (bool(SAM_API_KEY) or "fal.run" not in SAM_API_URL)


def build_sam_request(image_bgr: np.ndarray, text_prompt: str, point: Optional[tuple[int, int]]) -> dict:
    """Request body in fal.ai's fal-ai/sam-3/image schema.

    The image travels inline as a data URI (fal accepts those in `image_url`), `sync_mode` makes
    the masks come back inline too, and `apply_mask=False` asks for plain binary masks rather
    than a cut-out preview. Point prompts are in the pixel space of the image we send.

    Observed on fal: with a text prompt every instance of the concept comes back as its own mask
    (a house roof arrives as 5-10 sections); a point prompt is honoured only when `prompt` is
    the empty string, and then returns the object under the point.
    """
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(500, "Could not encode image for the SAM API.")
    body: dict = {
        "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii"),
        "prompt": text_prompt,          # "" for a point-only call: fal ignores points when text is given
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": SAM_MAX_MASKS,
        "include_scores": True,
        "include_boxes": True,
        "sync_mode": True,
        "output_format": "png",
    }
    if point is not None:
        body["point_prompts"] = [{"x": int(point[0]), "y": int(point[1]), "label": 1}]
    return body


def http_json(method: str, url: str, body: Optional[dict] = None) -> object:
    headers = {"Accept": "application/json"}
    if SAM_API_KEY:
        headers["Authorization"] = f"{SAM_API_AUTH} {SAM_API_KEY}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=SAM_API_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", "replace")[:300]
        raise HTTPException(502, f"SAM API returned HTTP {err.code}: {detail}")
    except urllib.error.URLError as err:
        raise HTTPException(502, f"SAM API unreachable: {err.reason}")
    except (TimeoutError, json.JSONDecodeError) as err:
        raise HTTPException(502, f"SAM API bad response: {err}")


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
                raise HTTPException(502, f"SAM API job {state.lower()}: {str(status)[:300]}")
            time.sleep(0.75)
        raise HTTPException(504, "SAM API job did not finish in time.")
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
    masks = []
    for item in items:
        url = item.get("url") if isinstance(item, dict) else item
        mask = decode_image_string(url) if isinstance(url, str) else None
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


def sam3_text_instances(image_bgr: np.ndarray, text_prompt: str) -> list[dict]:
    """Every instance of the concept, e.g. each roof section for "roof"."""
    payload = call_sam_api(build_sam_request(image_bgr, text_prompt, None))
    shape = image_bgr.shape[:2]
    masks = masks_from_fal(payload, shape) or masks_from_response(payload, shape)
    instances = []
    for mask, score in zip(masks, sam3_scores(payload, len(masks))):
        inst = make_instance(mask, "text", score)
        if inst is not None:
            instances.append(inst)
    return instances


def sam3_point_instance(image_bgr: np.ndarray, point: tuple[int, int]) -> Optional[dict]:
    """The object under one positive point (a section the text prompt missed)."""
    payload = call_sam_api(build_sam_request(image_bgr, "", point))
    shape = image_bgr.shape[:2]
    masks = masks_from_fal(payload, shape) or masks_from_response(payload, shape)
    if not masks:
        return None
    scores = sam3_scores(payload, len(masks))
    x, y = int(point[0]), int(point[1])
    ranked = sorted(zip(masks, scores), key=lambda ms: (bool(ms[0][y, x]) if 0 <= y < shape[0] and 0 <= x < shape[1] else False, ms[1] or 0), reverse=True)
    mask, score = ranked[0]
    return make_instance(mask, "point", score)


# --------------------------------------------------------------------------------------------
# Segmentation: local strategies (no API key needed)
# --------------------------------------------------------------------------------------------

def segment_mock(image_bgr: np.ndarray, point: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Local stand-in for a SAM point-prompt call, using GrabCut initialised from the point:
      * a thin band along the image border    -> definite background
      * everything else                       -> probable background
      * a box around the prompt               -> probable foreground (generous for the image
                                                 centre, tighter around an explicit click so
                                                 one click means one roof section)
      * a small disc around the prompt        -> definite foreground
    Nothing is asserted from colour (no "sky" rule): pale walls and overcast skies are too
    similar for that to be safe, and GrabCut's own colour models handle it well enough.
    Returns a uint8 mask (0/255) at the input resolution.
    """
    H, W = image_bgr.shape[:2]
    scale = min(1.0, GRABCUT_MAX_DIM / max(H, W))
    small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr
    h, w = small.shape[:2]

    if point is None:
        px, py = w // 2, int(h * 0.45)
    else:
        px = int(np.clip(point[0] * scale, 0, w - 1))
        py = int(np.clip(point[1] * scale, 0, h - 1))

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    # Probable foreground box around the prompt, biased upward (roofs sit above the click).
    bw, bh_up, bh_down = (0.42, 0.42, 0.36) if point is None else (0.2, 0.16, 0.12)
    x0, x1 = max(0, px - int(w * bw)), min(w, px + int(w * bw))
    y0, y1 = max(0, py - int(h * bh_up)), min(h, py + int(h * bh_down))
    gc[y0:y1, x0:x1] = cv2.GC_PR_FGD

    border = max(2, int(min(h, w) * 0.02))
    gc[:border, :] = cv2.GC_BGD
    gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = cv2.GC_BGD
    gc[:, -border:] = cv2.GC_BGD

    cv2.circle(gc, (px, py), max(4, int(min(h, w) * 0.03)), int(cv2.GC_FGD), thickness=cv2.FILLED)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, gc, None, bgd_model, fgd_model, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # Degenerate init (e.g. no background pixels left). Fall back to the plain threshold.
        return segment_threshold(image_bgr, 80, invert=True)

    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    k = max(3, int(min(h, w) * 0.012) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))   # bridge small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # drop 1-2 px slivers
    mask = keep_component_at(mask, (px, py))
    mask = fill_holes(mask)

    if scale < 1:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def segment_threshold(image_bgr: np.ndarray, threshold: int = 80, invert: bool = True) -> np.ndarray:
    """Deterministic fallback: blur -> grayscale threshold -> tidy -> largest blob -> fill holes.

    With `invert=True` (the default) dark pixels become foreground, which suits the common case
    of a dark roof against a bright sky.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, int(threshold), 255, mode)

    k = max(3, int(min(mask.shape[:2]) * 0.006) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = keep_component_at(mask, None)
    return fill_holes(mask)


# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------

HORIZONTAL_DEG = 12.0   # |angle| at or below this counts as a horizontal edge (ridge / eave)
VERTICAL_DEG = 78.0     # |angle| at or above this counts as a vertical edge (wall)
PITCH_MAX_DEG = 75.0    # steeper than this and "rise per 12" stops being meaningful
END_FIT_MED_PX = 1.5    # a fitted line is trusted at a corner only if the boundary pixels there sit this close to it


def map_vertices_to_contour(contour: np.ndarray, approx: np.ndarray) -> Optional[list[int]]:
    """Index of each approxPolyDP vertex inside the dense contour, following contour order."""
    lookup: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(contour):
        lookup.setdefault((int(x), int(y)), []).append(i)
    indices, previous = [], -1
    for x, y in approx:
        candidates = lookup.get((int(x), int(y)))
        if not candidates:
            return None
        ahead = [c for c in candidates if c > previous]
        index = ahead[0] if ahead else candidates[0]
        indices.append(index)
        previous = index
    return indices


def fit_edge(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust (Huber) total-least-squares line through the boundary pixels of one edge.

    Returns (origin, unit direction) with the direction oriented along the contour traversal.
    """
    first, last = points[0], points[-1]
    if len(points) < 3:
        direction = last - first
        norm = np.linalg.norm(direction)
        return first.copy(), (direction / norm if norm else np.array([1.0, 0.0]))
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    direction = np.array([vx, vy], float)
    if np.dot(direction, last - first) < 0:
        direction = -direction
    return np.array([x0, y0], float), direction


def intersect_lines(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((o2[0] - o1[0]) * d2[1] - (o2[1] - o1[1]) * d2[0]) / cross
    return o1 + t * d1


def refine_polygon(contour: np.ndarray, approx: np.ndarray, merge_angle_deg: float, guard_px: float) -> Optional[np.ndarray]:
    """Sub-pixel polygon from a Douglas-Peucker skeleton.

    approxPolyDP decides *where the corners are* but its vertices are stuck on boundary pixels,
    so the slope of an edge is only as good as its two endpoints. Here every edge is refit by
    least squares to all the boundary pixels it spans, adjacent edges that are nearly collinear
    (directed angle below `merge_angle_deg`) are merged, and each vertex becomes the intersection
    of its two fitted lines.

    An intersection is only used when the outline really passes through it (some boundary pixel
    of the two edges lies within `reach` of it) and both fitted lines fit the boundary pixels
    next to it (median residual at most END_FIT_MED_PX). That tells a long edge with a bump in
    the middle (fine: the line is right at both ends) apart from a Douglas-Peucker edge that
    swallowed a small L-shaped feature (its line is wrong at that end). Otherwise, if exactly one
    of the two lines fits at the corner pixel, the corner is projected onto that line so the
    trusted edge keeps its slope; if neither does (or they are parallel, i.e. a spike), the
    corner pixel is kept as is.
    """
    indices = map_vertices_to_contour(contour, approx)
    if indices is None or len(indices) < 3:
        return None

    runs: list[np.ndarray] = []
    for k in range(len(indices)):
        a, b = indices[k], indices[(k + 1) % len(indices)]
        run = contour[a:b + 1] if b >= a else np.concatenate([contour[a:], contour[:b + 1]])
        runs.append(run.astype(float))
    lines = [fit_edge(run) for run in runs]

    merge_cos = math.cos(math.radians(merge_angle_deg))
    while len(runs) > 3:
        best_cos, best_k = merge_cos, -1
        for k in range(len(runs)):
            cos = float(np.dot(lines[k][1], lines[(k + 1) % len(runs)][1]))
            if cos > best_cos:
                best_cos, best_k = cos, k
        if best_k < 0:
            break
        j = (best_k + 1) % len(runs)
        runs[best_k] = np.concatenate([runs[best_k], runs[j][1:]])
        lines[best_k] = fit_edge(runs[best_k])
        del runs[j], lines[j]

    window = max(6.0, min(guard_px, 30.0))     # px of boundary inspected on each side of a corner
    reach = max(3.0, 0.25 * guard_px)          # how far an intersection may sit from any boundary pixel

    def fits_near(run: np.ndarray, line: tuple[np.ndarray, np.ndarray], t_center: float, after: bool) -> bool:
        origin, direction = line
        normal = np.array([-direction[1], direction[0]])
        t = (run - origin) @ direction
        chosen = (t >= t_center - 1) & (t <= t_center + window) if after else (t <= t_center + 1) & (t >= t_center - window)
        if int(chosen.sum()) < 3:
            return False
        return float(np.median(np.abs((run[chosen] - origin) @ normal))) <= END_FIT_MED_PX

    def project(point: np.ndarray, line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        origin, direction = line
        return origin + ((point - origin) @ direction) * direction

    n = len(runs)
    vertices = []
    for k in range(n):
        run_a, line_a = runs[k - 1], lines[k - 1]     # edge ending at this corner
        run_b, line_b = runs[k], lines[k]             # edge starting at this corner
        corner = run_b[0]
        point = intersect_lines(line_a[0], line_a[1], line_b[0], line_b[1])
        if point is not None and np.linalg.norm(point - corner) <= guard_px:
            nearby = np.concatenate([run_a[-int(window) - int(reach) - 2:], run_b[:int(window) + int(reach) + 2]])
            reached = float(np.min(np.linalg.norm(nearby - point, axis=1))) <= reach
            t_a = float((point - line_a[0]) @ line_a[1])
            t_b = float((point - line_b[0]) @ line_b[1])
            if reached and fits_near(run_a, line_a, t_a, after=False) and fits_near(run_b, line_b, t_b, after=True):
                vertices.append(point)
                continue
        ok_a = fits_near(run_a, line_a, float((corner - line_a[0]) @ line_a[1]), after=False)
        ok_b = fits_near(run_b, line_b, float((corner - line_b[0]) @ line_b[1]), after=True)
        if ok_a != ok_b:
            vertices.append(project(corner, line_a if ok_a else line_b))
        else:
            vertices.append(corner)
    return np.array(vertices, float)


def polygon_is_ccw(polygon: np.ndarray) -> bool:
    """Shoelace sign. Positive area means the interior lies to the left of each directed edge."""
    x, y = polygon[:, 0], polygon[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) > 0


def describe_segment(p1: np.ndarray, p2: np.ndarray, interior_below: bool, bbox: tuple[int, int, int, int]) -> dict:
    """Slope, angle, roof pitch, length and a heuristic role for one straight roof vector.

    Points are pixel coordinates (origin top-left, y down). `slope` and `angle_deg` follow the
    on-screen reading of the line (rising to the right is positive), i.e. slope = -(dy/dx).
    `slope_pixel` is the raw dy/dx in image coordinates for anyone who needs it.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if x2 < x1 or (x2 == x1 and y2 < y1):    # normalise so `run` is never negative
        x1, y1, x2, y2 = x2, y2, x1, y1
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(-dy, dx)) if dx or dy else 0.0
    vertical = abs(angle) >= 89.9

    if vertical:
        slope = slope_pixel = None
        pitch = None
    else:
        slope_pixel = dy / dx
        slope = -slope_pixel
        pitch = f"{abs(slope) * 12:.1f}:12" if abs(angle) <= PITCH_MAX_DEG else None

    _, y0, _, h = bbox
    if abs(angle) <= HORIZONTAL_DEG:
        role = "ridge" if interior_below else "eave"
    elif abs(angle) >= VERTICAL_DEG:
        role = "wall"
    else:
        role = "slope"

    return {
        "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
        "slope": None if slope is None else round(slope, 5) + 0.0,           # + 0.0 turns -0.0 into 0.0
        "slope_pixel": None if slope_pixel is None else round(slope_pixel, 5) + 0.0,
        "vertical": vertical,
        "angle_deg": round(angle, 2) + 0.0,
        "pitch": pitch,
        "length_px": round(length, 2),
        "role": role,
        "mid_y": (y1 + y2) / 2.0,      # used for the cutoff filter, dropped before returning
        "rel_y": ((y1 + y2) / 2.0 - y0) / h if h else 0.0,
    }


def weighted_median(values: list[float], weights: list[float]) -> Optional[float]:
    if not values:
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(np.array(weights, float)[order])
    return float(np.array(values)[order][int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def summarise(segments: list[dict], bbox: tuple[int, int, int, int]) -> dict:
    """Roof-level numbers: dominant pitch from the sloped edges, ridge and eave heights."""
    slopes = [s for s in segments if s["role"] == "slope" and s["slope"] is not None]
    dominant = weighted_median([abs(s["slope"]) for s in slopes], [s["length_px"] for s in slopes])
    # Ridge height: the horizontal ridge if there is one, else the apex of the sloped edges
    # (a gable seen from the front has no horizontal ridge in its silhouette).
    ridges = [s for s in segments if s["role"] == "ridge"]
    eaves = [s for s in segments if s["role"] == "eave"]
    ridge_y = min(s["mid_y"] for s in ridges) if ridges else (min(p[1] for s in slopes for p in s["points"]) if slopes else None)
    eave_y = max(s["mid_y"] for s in eaves) if eaves else None
    return {
        "slope_segments": len(slopes),
        "dominant_slope": None if dominant is None else round(dominant, 5),
        "dominant_angle_deg": None if dominant is None else round(math.degrees(math.atan(dominant)), 2),
        "dominant_pitch": None if dominant is None else f"{dominant * 12:.1f}:12",
        "ridge_y_px": None if ridge_y is None else round(ridge_y, 2),
        "eave_y_px": None if eave_y is None else round(eave_y, 2),
        "rise_px": None if ridge_y is None or eave_y is None else round(eave_y - ridge_y, 2),
        "span_px": int(bbox[2]),
    }


def merge_shared_edges(edges: list[dict], tol_px: float, angle_deg: float = 5.0) -> list[dict]:
    """Two roof sections meet along a hip, valley or ridge, so each section's polygon carries its
    own copy of that edge, a few pixels apart. Edges from different sections that are collinear
    (direction within `angle_deg`, both endpoints within `tol_px` of the other's line) and overlap
    along their length are merged into one edge spanning both.
    """
    cos_limit = math.cos(math.radians(angle_deg))
    kept: list[dict] = []
    for edge in sorted(edges, key=lambda e: -np.linalg.norm(e["p2"] - e["p1"])):
        p1, p2 = edge["p1"], edge["p2"]
        merged = False
        for base in kept:
            if base["instance"] == edge["instance"]:
                continue
            b1, b2 = base["p1"], base["p2"]
            direction = b2 - b1
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            direction /= length
            normal = np.array([-direction[1], direction[0]])
            e_dir = p2 - p1
            e_len = float(np.linalg.norm(e_dir))
            if e_len < 1e-6 or abs(float(np.dot(direction, e_dir / e_len))) < cos_limit:
                continue
            if abs(float(np.dot(p1 - b1, normal))) > tol_px or abs(float(np.dot(p2 - b1, normal))) > tol_px:
                continue
            t1, t2 = sorted((float(np.dot(p1 - b1, direction)), float(np.dot(p2 - b1, direction))))
            overlap = min(length, t2) - max(0.0, t1)
            if overlap < 0.3 * min(length, e_len):
                continue
            lo, hi = min(0.0, t1), max(length, t2)
            base["p1"], base["p2"] = b1 + lo * direction, b1 + hi * direction
            base["shared_with"].append(edge["instance"])
            merged = True
            break
        if not merged:
            edge["shared_with"] = []
            kept.append(edge)
    return kept


def extract_roofline(contours: list[tuple[int, np.ndarray]], epsilon: float, height_cutoff: float,
                     refine: bool = True, merge_angle_deg: float = 5.0) -> dict:
    """Vectorise every section outline and keep the segments above the height cutoff.

    contours       (instance index, dense contour) pairs, one or more per roof section.
    epsilon        Fraction of the *largest* contour's perimeter, used as the approxPolyDP
                   tolerance in pixels for every section, so small sections are simplified to
                   the same physical tolerance as the main roof instead of following every wobble.
    height_cutoff  Percentage of the combined bounding-box height. 0 keeps every segment;
                   at 60 the lower 60 % (gutters, walls, foundation) is dropped. A segment is
                   kept when its midpoint lies above the cutoff line.
    refine         Least-squares refit + line-intersection vertices (see refine_polygon).
    """
    xs = np.concatenate([c[:, 0, 0] for _, c in contours])
    ys = np.concatenate([c[:, 0, 1] for _, c in contours])
    x0, y0 = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    cutoff_y = y0 + h * (1.0 - height_cutoff / 100.0)

    reference_perimeter = max(cv2.arcLength(c, True) for _, c in contours)
    eps_px = epsilon * reference_perimeter

    edges: list[dict] = []
    contour_meta: list[dict] = []
    for cid, (instance, contour) in enumerate(contours):
        perimeter = cv2.arcLength(contour, True)
        # A tiny section still needs a sane polygon: never let its tolerance exceed 4 % of its own outline.
        own_eps = min(eps_px, 0.04 * perimeter)
        approx = cv2.approxPolyDP(contour, own_eps, True).reshape(-1, 2)
        dense = contour.reshape(-1, 2)
        polygon = None
        if refine and len(approx) >= 3:
            polygon = refine_polygon(dense, approx, merge_angle_deg, guard_px=min(max(8.0, own_eps), 0.1 * max(w, h)))
        if polygon is None:
            polygon = approx.astype(float)
        if len(polygon) < 3:
            continue

        bbox = cv2.boundingRect(contour)
        ccw = polygon_is_ccw(polygon)
        n = len(polygon)
        for i in range(n):
            p1, p2 = polygon[i], polygon[(i + 1) % n]
            d = p2 - p1
            inward = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
            edges.append({"p1": p1.copy(), "p2": p2.copy(), "interior_below": bool(inward[1] > 0),
                          "bbox": bbox, "instance": instance, "contour": cid})

        contour_meta.append({
            "id": cid,
            "instance": instance,
            "area_px": round(float(cv2.contourArea(contour)), 1),
            "perimeter_px": round(float(perimeter), 1),
            "bounding_box": {"x": int(bbox[0]), "y": int(bbox[1]), "width": int(bbox[2]), "height": int(bbox[3])},
            "epsilon_px": round(float(own_eps), 2),
            "vertex_count": int(n),
            "douglas_peucker_vertices": int(len(approx)),
        })

    if len({inst for inst, _ in contours}) > 1:
        edges = merge_shared_edges(edges, tol_px=max(4.0, 0.006 * max(w, h)))

    segments: list[dict] = []
    for edge in edges:
        seg = describe_segment(edge["p1"], edge["p2"], interior_below=edge["interior_below"], bbox=edge["bbox"])
        if seg["mid_y"] > cutoff_y:
            continue
        seg["contour"] = edge["contour"]
        seg["instance"] = edge["instance"]
        seg["shared_with"] = sorted(set(edge.get("shared_with", [])))
        segments.append(seg)

    # A "base" is the lowest eave of a silhouette that also has proper walls (house, not roof-only).
    meta_by_id = {m["id"]: m for m in contour_meta}
    for cid in {s["contour"] for s in segments}:
        mine = [s for s in segments if s["contour"] == cid]
        has_walls = any(s["role"] == "wall" and s["length_px"] >= max(0.3 * meta_by_id[cid]["bounding_box"]["height"], 0.15 * h) for s in mine)
        if has_walls:
            for s in mine:
                if s["role"] == "eave" and s["rel_y"] >= 0.9:
                    s["role"] = "base"

    summary = summarise(segments, (x0, y0, w, h))
    for i, s in enumerate(segments):
        s["id"] = i
        del s["mid_y"], s["rel_y"]
    vertices = sorted({tuple(pt) for s in segments for pt in s["points"]})

    return {
        "segments": segments,
        "vertices": [list(v) for v in vertices],
        "contours": contour_meta,
        "polygon_vertex_count": sum(c["vertex_count"] for c in contour_meta),
        "bounding_box": {"x": x0, "y": y0, "width": w, "height": h},
        "height_cutoff": {"percent": height_cutoff, "y_px": round(cutoff_y, 1)},
        "epsilon": {"fraction": epsilon, "pixels": round(float(eps_px), 2), "reference_perimeter_px": round(float(reference_perimeter), 1)},
        "refined": refine,
        "summary": summary,
    }


# --------------------------------------------------------------------------------------------
# Sample image so the app can be tried without hunting for a photo
# --------------------------------------------------------------------------------------------

def make_sample_house(w: int = 1200, h: int = 800) -> np.ndarray:
    """A synthetic gable house with a dormer and chimney, sky above, lawn below."""
    rng = np.random.default_rng(7)
    img = np.zeros((h, w, 3), np.uint8)

    horizon = int(h * 0.70)
    t = np.linspace(0, 1, horizon)[:, None, None]
    sky_top, sky_bottom = np.array([232, 178, 118]), np.array([246, 226, 206])  # BGR
    img[:horizon] = (sky_top * (1 - t) + sky_bottom * t).astype(np.uint8)
    t = np.linspace(0, 1, h - horizon)[:, None, None]
    lawn_top, lawn_bottom = np.array([70, 150, 78]), np.array([50, 110, 55])
    img[horizon:] = (lawn_top * (1 - t) + lawn_bottom * t).astype(np.uint8)

    # Chimney (drawn first so the roof overlaps its base)
    cv2.rectangle(img, (760, 235), (804, 330), (70, 74, 120), cv2.FILLED)
    # Walls
    cv2.rectangle(img, (300, 385), (900, 640), (170, 200, 220), cv2.FILLED)
    # Main gable roof with overhang
    roof = np.array([[268, 388], [600, 198], [932, 388], [932, 402], [268, 402]], np.int32)
    cv2.fillPoly(img, [roof], (68, 58, 74))
    # Dormer on the left slope
    cv2.rectangle(img, (432, 296), (508, 372), (160, 190, 210), cv2.FILLED)
    cv2.fillPoly(img, [np.array([[420, 300], [470, 252], [520, 300]], np.int32)], (60, 52, 66))
    cv2.rectangle(img, (452, 316), (488, 356), (120, 90, 60), cv2.FILLED)
    # Door and windows
    cv2.rectangle(img, (570, 520), (630, 640), (62, 78, 118), cv2.FILLED)
    for x0 in (360, 460, 700, 800):
        cv2.rectangle(img, (x0, 440), (x0 + 60, 520), (140, 100, 70), cv2.FILLED)
        cv2.rectangle(img, (x0 + 6, 446), (x0 + 54, 514), (230, 200, 170), cv2.FILLED)

    noise = rng.normal(0, 5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


# --------------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    session_id: str
    mode: str = Field("mock", pattern="^(sam3|mock|threshold)$")
    threshold: int = Field(80, ge=0, le=255)
    invert: bool = True
    points: list[list[int]] = Field(default_factory=list, max_length=24)   # [[x, y], ...] one per section
    text_prompt: str = Field(SAM_TEXT_PROMPT, max_length=200)


class GeometryRequest(BaseModel):
    session_id: str
    epsilon: float = Field(0.01, ge=0.0005, le=0.1)
    height_cutoff: float = Field(0.0, ge=0.0, le=100.0)
    refine: bool = True
    merge_angle_deg: float = Field(5.0, ge=0.0, le=30.0)


def session_info(session_id: str, session: Session) -> dict:
    h, w = session.image.shape[:2]
    return {
        "session_id": session_id,
        "filename": session.filename,
        "width": w,
        "height": h,
        "source_width": session.source_size[0],
        "source_height": session.source_size[1],
        "scale_from_source": round(w / session.source_size[0], 6),
        "image_url": f"/api/image/{session_id}",
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/config")
def config() -> dict:
    parts = urllib.parse.urlsplit(SAM_API_URL) if SAM_API_URL else None
    return {
        "sam_api": sam_api_configured(),
        "sam_api_host": parts.netloc if parts else None,
        "sam_model": parts.path.strip("/") if parts else None,
        "sam_key_missing": bool(SAM_API_URL) and not SAM_API_KEY,
        "text_prompt": SAM_TEXT_PROMPT,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    image = decode_image(data)
    source_size = (image.shape[1], image.shape[0])
    session = Session(fit_to_working_size(image), source_size, file.filename or "upload")
    return session_info(put_session(session), session)


@app.post("/api/sample")
def sample() -> dict:
    image = make_sample_house()
    session = Session(image, (image.shape[1], image.shape[0]), "sample-house.png")
    return session_info(put_session(session), session)


@app.get("/api/image/{session_id}")
def image(session_id: str) -> Response:
    return Response(get_session(session_id).encoded, media_type="image/jpeg")


def cached_instances(session: Session, key: str, produce) -> list[dict]:
    """Prompts are cached per session so adding a click re-uses every earlier result."""
    if key not in session.cache:
        result = produce()
        session.cache[key] = [r for r in (result if isinstance(result, list) else [result]) if r is not None]
    return [dict(inst) for inst in session.cache[key]]


@app.post("/api/segment")
def segment(req: SegmentRequest) -> dict:
    session = get_session(req.session_id)
    started = time.perf_counter()
    points = [(int(p[0]), int(p[1])) for p in req.points if len(p) == 2]
    text = req.text_prompt.strip()
    instances: list[dict] = []

    if req.mode == "sam3":
        if not sam_api_configured():
            raise HTTPException(409, "SAM 3 is not configured. Put FAL_KEY in .env and restart the server.")
        if text:
            instances += cached_instances(session, f"sam3:text:{text}", lambda: sam3_text_instances(session.image, text))
        for point in points:
            instances += cached_instances(session, f"sam3:point:{point[0]},{point[1]}", lambda p=point: sam3_point_instance(session.image, p))
        if not text and not points:
            raise HTTPException(400, "Give SAM 3 a text prompt or click at least one roof section.")
    elif req.mode == "mock":
        if points:
            for point in points:
                instances += cached_instances(session, f"mock:point:{point[0]},{point[1]}", lambda p=point: make_instance(segment_mock(session.image, p), "point"))
        else:
            instances += cached_instances(session, "mock:centre", lambda: make_instance(segment_mock(session.image, None), "centre"))
    else:
        instances += [i for i in [make_instance(segment_threshold(session.image, req.threshold, req.invert), "threshold")] if i]

    for i, inst in enumerate(instances):
        inst["color"] = "#%02x%02x%02x" % instance_color(i)[::-1]
    session.instances = instances
    session.contours = [(i, c) for i, inst in enumerate(instances) for c in usable_contours(inst["mask"])]
    session.segmentation = req.model_dump()

    shape = session.image.shape[:2]
    union = np.zeros(shape, np.uint8)
    for inst in instances:
        union = cv2.bitwise_or(union, inst["mask"])
    return {
        "mask": encode_png_base64(render_instances(instances, shape) if instances else union),
        "instances": [{"id": i, "source": inst["source"], "score": inst["score"], "color": inst["color"],
                       "area_percent": round(float(np.count_nonzero(inst["mask"])) / union.size * 100, 2)}
                      for i, inst in enumerate(instances)],
        "coverage_percent": round(float(np.count_nonzero(union)) / union.size * 100, 2),
        "contour_count": len(session.contours),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/api/geometry")
def geometry(req: GeometryRequest) -> dict:
    session = get_session(req.session_id)
    if not session.segmentation:
        raise HTTPException(409, "Run segmentation before requesting geometry.")
    started = time.perf_counter()

    if not session.contours:
        result = {"segments": [], "vertices": [], "contours": [], "polygon_vertex_count": 0,
                  "summary": summarise([], (0, 0, 0, 0)), "refined": req.refine}
    else:
        result = extract_roofline(session.contours, req.epsilon, req.height_cutoff, req.refine, req.merge_angle_deg)

    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
