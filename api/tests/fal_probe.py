"""Day 0 gate — verify the fal.ai SAM 3 contract before anything depends on it.

Every call is billed pay-as-you-go, so this is deliberately frugal:

    python tests/fal_probe.py                      # 1 call: text prompt, resolution, scores, boxes
    python tests/fal_probe.py --full               # 3 calls: adds box and point prompts
    python tests/fal_probe.py [--full] photo.png   # any real house photo

It must run on a photo that actually contains a roof. An earlier version picked the first
image alphabetically — an architecture diagram — and spent calls reporting "no masks".

fal's model page advertises a fixed 1024x1024 mask output while the SAM 3 model card says
masks match the input. `routers/regions.py` derives roofline polylines from `mask.shape`
with no rescaling, so the resolution check is the one that gates Day 1. It is measured on
every returned mask, from every call, at the working-copy size production actually posts.
"""

import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_PHOTO = (
    Path.home() / "Documents/freelance/Hammad-project/bulb-project/lighting project/house02.png"
)

results: list[tuple[str, str, str]] = []
billed = 0


def record(name: str, verdict: str, detail: str) -> None:
    results.append((name, verdict, detail))
    print(f"  {verdict:<8} {name}\n           {detail}\n", flush=True)


def load_key() -> str:
    """Environment first, then api/.env — fal_client only ever reads os.environ."""
    if key := os.getenv("FAL_KEY"):
        return key
    try:
        from app.config import get_settings

        key = get_settings().fal_key
    except Exception as exc:  # noqa: BLE001
        print(f"could not read api/.env: {type(exc).__name__}: {exc}")
        return ""
    if key:
        os.environ["FAL_KEY"] = key
    return key


def main() -> int:
    global billed
    args = [a for a in sys.argv[1:] if a != "--full"]
    full = "--full" in sys.argv[1:]
    photo = Path(args[0]) if args else DEFAULT_PHOTO

    if not load_key():
        print(
            "FAL_KEY is not set.\n"
            "  Put FAL_KEY=<key> in api/.env and run from the api/ directory, or export it."
        )
        return 2
    if not photo.exists():
        print(f"Photo not found: {photo}")
        return 2

    import fal_client
    import httpx

    from app.config import get_settings
    from app.providers.fal import _data_uri
    from app.services import detect

    settings = get_settings()
    image = Image.open(photo).convert("RGB")
    small, _ = detect.working_copy(image, settings.detect_max_edge)
    posted = (small.width, small.height)
    planned = 3 if full else 1
    print(f"Probing {settings.fal_model} with {photo.name}, posted at {posted[0]}x{posted[1]}")
    print(f"This spends {planned} billed call(s).\n")

    shapes: set[tuple[int, int]] = set()

    def call(prompt: str, **extra) -> dict:
        global billed
        arguments = {
            "image_url": _data_uri(small),
            "prompt": prompt,
            "apply_mask": False,
            "include_scores": True,
            "include_boxes": True,
            "output_format": "png",
            **extra,
        }
        started = time.time()
        billed += 1
        output = fal_client.subscribe(settings.fal_model, arguments=arguments, client_timeout=180)
        for item in output.get("masks") or []:
            data = httpx.get(item["url"], timeout=120).content  # CDN fetch, not a billed call
            mask = np.array(Image.open(io.BytesIO(data)).convert("L"))
            shapes.add((mask.shape[1], mask.shape[0]))
        print(f"  (call {billed} took {time.time() - started:.1f}s)\n")
        return output

    # 1 — text prompt: data URI accepted, every instance returned, scores and boxes present.
    try:
        out = call("roof", return_multiple_masks=True, max_masks=settings.fal_max_masks)
    except Exception as exc:  # noqa: BLE001
        record("text prompt call", "FAIL", f"{type(exc).__name__}: {str(exc)[:300]}")
        return 1

    masks, scores, boxes = out.get("masks") or [], out.get("scores") or [], out.get("boxes") or []
    record("data URI accepted as image_url", "PASS", "call succeeded without uploading the photo")
    record(
        "text prompt returns multiple instances",
        "PASS" if len(masks) > 1 else ("UNKNOWN" if masks else "FAIL"),
        f"{len(masks)} masks for 'roof'",
    )
    record(
        "scores returned",
        "PASS" if len(scores) == len(masks) and masks else "UNKNOWN",
        f"{[round(float(s), 3) for s in scores]}",
    )
    record(
        "boxes are normalised [cx, cy, w, h]",
        "PASS" if boxes and all(0.0 <= float(v) <= 1.0 for v in boxes[0]) else "UNKNOWN",
        f"first box = {boxes[0] if boxes else None}",
    )

    # 2 — box and point prompts, which back the editor's re-segment and click-to-select.
    if full:
        cx, cy = small.width // 2, small.height // 2
        for name, extra in (
            ("box prompt", {"box_prompts": [{"x_min": cx - 120, "y_min": cy - 90,
                                             "x_max": cx + 120, "y_max": cy + 90}]}),
            ("point prompt", {"point_prompts": [{"x": cx, "y": cy, "label": 1}]}),
        ):
            try:
                result = call("", return_multiple_masks=False, **extra)
                count = len(result.get("masks") or [])
                record(f"{name} with an empty text prompt", "PASS" if count else "FAIL",
                       f"{count} mask(s)")
            except Exception as exc:  # noqa: BLE001
                record(f"{name} with an empty text prompt", "FAIL",
                       f"{type(exc).__name__}: {str(exc)[:200]}")

    # 3 — the gate: every mask from every call must be in the posted frame.
    if not shapes:
        record("mask resolution equals posted resolution", "UNKNOWN", "no masks to measure")
    else:
        record(
            "mask resolution equals posted resolution",
            "PASS" if shapes == {posted} else "FAIL",
            f"posted {posted[0]}x{posted[1]}; returned {sorted(shapes)}",
        )

    print("=" * 72)
    for name, verdict, _ in results:
        print(f"{verdict:<8} {name}")
    failures = sum(1 for _, v, _ in results if v == "FAIL")
    print("=" * 72)
    print(f"{billed} billed call(s). {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
