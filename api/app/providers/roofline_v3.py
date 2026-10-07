"""Direct typed roof lines, adapted from the supplied roofline-v3-sol-cli.

The shipped V3 prompt, image encoding, schema, model and reasoning defaults
are preserved. detect() answers in the pixel frame of the image it is given; detect_crop()
gives it the target-house crop and caches the answer. Mapping crop coordinates back onto the
photo is services/rooflines.crop_to_image, never done here.
"""

import base64
import hashlib
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from app.config import Settings

API = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "roofline_v3.txt"
TYPES = ("eave", "rake", "hip", "ridge", "valley", "junction", "unknown")
# Sent to OpenRouter verbatim as the model's structured-output contract (strict json_schema).
# Kept as a plain dict, not generated from the models below: the provider's strict mode has its
# own constraints (every property required, no defs indirection) that need not match how the
# response is validated once it comes back.
SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": list(TYPES)},
            "points": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["type", "points"], "additionalProperties": False,
    }}},
    "required": ["lines"], "additionalProperties": False,
}


class _RawLine(BaseModel):
    """One line as the model returns it, before rescaling. Mirrors SCHEMA's item shape."""

    model_config = ConfigDict(extra="forbid")

    type: str
    # Strict: lax mode would silently accept a bool (a subclass of int) or a whole-number float
    # ("10.0") as a coordinate, which the original hand-written check rejected outright.
    points: list[StrictInt] = Field(min_length=4, max_length=4)

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in TYPES:
            raise ValueError(f"unknown line type {value!r}")
        return value


class _RawLines(BaseModel):
    """The model's whole answer, before rescaling. Mirrors SCHEMA's top level."""

    model_config = ConfigDict(extra="forbid")

    lines: list[_RawLine]


# The model is handed a working copy, not the upload. A phone or stock photo arrives at several
# thousand pixels; encoded losslessly for transmission that is tens of megabytes, and the request
# was simply refused -- "image exceeds the 10 MB encoded-image limit". The SAM path has always
# posted a 2000px copy for the same reason. Coordinates come back in the posted image's space and
# are mapped back here, so nothing downstream sees the difference.
UPLOAD_MAX_EDGE = 2000


def for_upload(image: Image.Image) -> tuple[Image.Image, float]:
    """The copy that is posted, and the factor taking its coordinates back to the original."""
    longest = max(image.size)
    if longest <= UPLOAD_MAX_EDGE:
        return image, 1.0
    scale = UPLOAD_MAX_EDGE / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS), image.width / size[0]


def rescale(lines: list[dict], factor: float, size: tuple[int, int]) -> list[dict]:
    """Points from the posted copy back onto the original, clamped inside it."""
    if factor == 1.0:
        return lines
    width, height = size
    for line in lines:
        x1, y1, x2, y2 = line["points"]
        line["points"] = [
            min(int(round(x1 * factor)), width - 1), min(int(round(y1 * factor)), height - 1),
            min(int(round(x2 * factor)), width - 1), min(int(round(y2 * factor)), height - 1),
        ]
    return lines


def image_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue())
    mime = "image/png"
    if len(encoded) > 10_000_000:
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", lossless=True, method=4)
        encoded = base64.b64encode(buffer.getvalue())
        mime = "image/webp"
    if len(encoded) > 10_000_000:
        raise RuntimeError("Roofline image exceeds the 10 MB encoded-image limit even after "
                           "downscaling; lower UPLOAD_MAX_EDGE")
    return f"data:{mime};base64,{encoded.decode()}"


def target_text(house: tuple[int, int, int, int] | None, size: tuple[int, int]) -> str:
    """The prompt's TARGET HOUSE paragraph: which building to draw, as a box in the posted image's
    pixels, or (no box) the house the image shows. V3 was told to draw "background" roofs and drew
    the neighbours: 34 of 71 false lines on 17 annotated photos."""
    if house is None:
        return ("Annotate every roof that belongs to the house this image shows, with all its "
                "parts. Do not annotate the roofs of other buildings.")
    width, height = size
    x0, y0, x1, y1 = house
    x0, x1 = min(max(int(x0), 0), width - 1), min(max(int(x1), 0), width - 1)
    y0, y1 = min(max(int(y0), 0), height - 1), min(max(int(y1), 0), height - 1)
    return (f"The house being quoted is the building whose roof lies within x {x0} to {x1} and "
            f"y {y0} to {y1} of this image. Annotate only that house's roof edges. Its edges may run "
            "a little past this box: follow them to where they really end. Do not annotate the roofs "
            "of neighbouring or background buildings, even where they are visible or touch it.")


def validate(data: dict, size: tuple[int, int]) -> list[dict]:
    # Structural shape (fields, types, the enum) is Pydantic's job; only the parts that depend
    # on this particular image -- in-bounds coordinates, no zero-length edge -- stay hand-checked.
    parsed = _RawLines.model_validate(data)
    lines: list[dict] = []
    for index, line in enumerate(parsed.lines, 1):
        if any(not 0 <= p < limit for p, limit in zip(line.points, size * 2)):
            raise ValueError(f"Line {index}: coordinate outside image")
        if line.points[:2] == line.points[2:]:
            raise ValueError(f"Line {index}: zero-length edge")
        lines.append({"type": line.type, "points": list(line.points)})
    return lines


# Bump when anything that shapes the request or the parsed answer changes outside the prompt
# file, model or reasoning (all three are in the key already): the schema, the posted size, the
# encoding. Old entries then miss rather than hand back an answer to a different question.
CACHE_VERSION = 1


def _digest(image: Image.Image) -> str:
    """Content hash of pixels and size: the same photo under another filename hits, a re-upload
    that changed a single pixel misses."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()


def cache_key(image: Image.Image, box: tuple[int, int, int, int], crop: Image.Image, settings: Settings,
              house: tuple[int, int, int, int] | None = None) -> str:
    """Deterministic key of a V3 request: whole photo, crop box and crop pixels, the target house's
    box, model, reasoning and prompt. Any change to the crop, house, prompt or model is a different key."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    parts = {
        "version": CACHE_VERSION,
        "image": _digest(image),
        "crop": _digest(crop),
        "box": [int(v) for v in box],
        "house": [int(v) for v in house] if house is not None else None,
        "model": settings.roofline_model,
        "reasoning": settings.roofline_reasoning,
        "prompt": hashlib.sha256(template.encode()).hexdigest(),
        "schema": hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
        "upload_max_edge": UPLOAD_MAX_EDGE,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def detect_crop(image: Image.Image, box: tuple[int, int, int, int], settings: Settings,
                cache_dir: Path | None = None,
                house: tuple[int, int, int, int] | None = None) -> tuple[list[dict], dict]:
    """V3 on one crop of the photo: lines in the crop's own pixel frame, and the run's metadata.

    `box` is (x0, y0, x1, y1) in photo pixels, x1/y1 exclusive; `house`, in photo pixels too, is
    the target house the model is told to draw (None: the house the crop shows). The answer is
    cached by cache_key when `cache_dir` is given: a repeated detection of the same crop returns
    the stored lines and makes no call. Only complete, validated answers are stored; a failure is
    never cached, so it does not stick.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = image.crop((x0, y0, x1, y1))
    inside = None if house is None else (int(house[0]) - x0, int(house[1]) - y0, int(house[2]) - x0, int(house[3]) - y0)
    key = cache_key(image, (x0, y0, x1, y1), crop, settings, inside)
    path = cache_dir / f"{key}.json" if cache_dir is not None else None
    if path is not None and path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            lines = validate({"lines": stored["lines"]}, crop.size)
            meta = {**stored["meta"], "cache": "hit", "cache_key": key, "crop_box": [x0, y0, x1, y1],
                    # Nothing was billed for this run; the original bill stays on record.
                    "cached_cost_usd": stored["meta"].get("cost_usd"), "cost_usd": 0.0, "elapsed_seconds": 0.0}
            return lines, meta
        except (OSError, ValueError, KeyError, TypeError):
            pass  # an unreadable entry is a miss, and is overwritten below
    lines, meta = detect(crop, settings, inside)
    meta = {**meta, "cache": "miss" if path is not None else "off", "cache_key": key, "crop_box": [x0, y0, x1, y1],
            "house_box": list(house) if house is not None else None}
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"lines": lines, "meta": meta}), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass  # a cache that cannot be written costs a later call, never this detection
    return lines, meta


def detect(image: Image.Image, settings: Settings,
           house: tuple[int, int, int, int] | None = None) -> tuple[list[dict], dict]:
    """V3 on `image`; `house` is the target house's box in `image`'s pixels (None: the house it shows)."""
    if not settings.openrouter_api_key.strip():
        raise RuntimeError("V3 roof detection requires OPENROUTER_API_KEY on the API server")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    # Everything the model is told, and everything it answers, is in the posted copy's space.
    posted, factor = for_upload(image)
    posted_house = None if house is None else tuple(int(round(v / factor)) for v in house)
    prompt = (template.replace("{width}", str(posted.width)).replace("{height}", str(posted.height))
              .replace("{target}", target_text(posted_house, posted.size)))
    payload = {
        "model": settings.roofline_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"TARGET: {posted.width} x {posted.height} pixels."},
                {"type": "image_url", "image_url": {"url": image_url(posted)}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "roof_edges_typed", "strict": True, "schema": SCHEMA,
        }},
        "reasoning": {"effort": settings.roofline_reasoning},
        "max_completion_tokens": 8000,
        "provider": {"require_parameters": True}, "stream": False,
    }
    started = time.perf_counter()
    try:
        # No retries: an ambiguous timeout must not automatically incur another bill.
        response = httpx.post(API, json=payload, headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
        }, timeout=settings.roofline_timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"OpenRouter roof detection failed (HTTP {exc.response.status_code})") from None
    except httpx.RequestError:
        raise RuntimeError("OpenRouter roof detection timed out or could not connect") from None
    try:
        raw = response.json()
        if raw.get("error"):
            raise ValueError("Provider returned an error")
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("Incomplete or refused response")
        message = choice["message"]
        if message.get("refusal"):
            raise ValueError("Model refused the request")
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") == "text")
        lines = rescale(validate(json.loads(content), posted.size), factor, image.size)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise RuntimeError("OpenRouter returned invalid or incomplete roof geometry") from None
    usage = raw.get("usage") or {}
    # Only billing fields, never response text, keys or image payloads, enter app metadata.
    return lines, {
        "detector": "v3_sol", "model": settings.roofline_model,
        "returned_model": raw.get("model"), "reasoning": settings.roofline_reasoning,
        "provider": "openrouter", "response_id": raw.get("id"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "image_size": list(image.size), "posted_size": list(posted.size),
        "line_count": len(lines),
        "prompt_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "prompt_used_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost") if key in usage},
        "cost_usd": usage.get("cost"),
    }
