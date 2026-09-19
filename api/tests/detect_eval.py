"""Per-class detection on the client's photos, through the production code path.

Every fal response is cached in out/fal_cache/, keyed by photo, prompt and settings, so
re-running after a filter or threshold change costs nothing. Detection is billed per call,
so the default is a dry run that only reports what a real run would spend:

    python tests/detect_eval.py                                     # dry run: billed calls needed
    python tests/detect_eval.py --spend                             # make only the uncached calls
    python tests/detect_eval.py --spend --prompt shrub="round shrub"  # try a prompt variant
    python tests/detect_eval.py --photos "image (3).png" "Sechtor 2.jpg"

Compared against the GroundingDINO + SAM 2 baseline in detection-results/summary.txt, on the
same photos, when that file is present.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

API = Path(__file__).resolve().parents[1]
REPO = API.parent
sys.path.insert(0, str(API))
os.environ.setdefault("SEGMENTATION_PROVIDER", "fal")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.providers import fal as F  # noqa: E402
from app.providers import prompts  # noqa: E402
from app.services import detect, roofline  # noqa: E402

PHOTOS = REPO.parent / "lighting project"
CACHE = REPO / "out" / "fal_cache"
OUT = REPO / "out" / "detect-eval"
BASELINE = REPO / "detection-results" / "summary.txt"
PRICE_PER_CALL = 0.005

# The photos the GroundingDINO + SAM 2 baseline was measured on (image (6).png is gone).
DEFAULT_PHOTOS = [
    "20141126_174846.jpg", "20141126_175610.jpg", "20141126_175627.jpg",
    "20141126_175634.jpg", "20151218_201413_001.jpg", "IMG-20241028-WA0009.jpg",
    "IMG-20250125-WA0009.jpg", "IMG_20221128_190400.jpg", "IMG_20221209_214134_012.jpg",
    "IMG_20251108_115325 (5).jpg", "Sechtor 2.jpg", "image (3).png", "image (5).png",
]


def cache_key(photo: str, size: tuple[int, int], prompt: str, extra: dict,
              max_masks: int | None = None, image_format: str = "png") -> str:
    settings = get_settings()
    material = [photo, list(size), prompt, extra, settings.fal_model, max_masks or settings.fal_max_masks]
    # A JPEG post is a different request; PNG keys stay as they were.
    if image_format != "png":
        material.append(image_format)
    return hashlib.sha1(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


class CachedFal(F.FalProvider):
    """The real provider, with every response kept on disk and spending made explicit."""

    def __init__(self, photo: str, spend: bool, max_masks: int | None = None, image_format: str = "png"):
        super().__init__(max_masks=max_masks, image_format=image_format)
        self.photo = photo
        self.spend = spend
        self.billed = 0
        self._lock = threading.Lock()

    def _call(self, image, prompt, **extra):
        key = cache_key(self.photo, image.size, prompt, extra, self.max_masks, self.image_format)
        entry = CACHE / f"{key}.json"
        if entry.exists():
            return json.loads(entry.read_text())
        if not self.spend:
            raise SystemExit(f"uncached call for {self.photo} / {prompt!r} without --spend")

        with self._lock:
            self.billed += 1
        result = super()._call(image, prompt, **extra)
        # fal's CDN links are not permanent, so keep the mask files themselves.
        for index, item in enumerate(result.get("masks") or []):
            if isinstance(item, dict) and item.get("url"):
                path = CACHE / f"{key}_{index}.png"
                path.write_bytes(httpx.get(item["url"], timeout=120).content)
                item["url"] = str(path)
        entry.write_text(json.dumps(result))
        return result


_remote_mask = F._mask_from_url


def _cached_mask(url: str, size: tuple[int, int], timeout: float):
    path = Path(url)
    if path.is_file():
        mask = np.array(Image.open(path).convert("L")) > F.MASK_THRESHOLD
        return F._conform(mask, size)
    return _remote_mask(url, size, timeout)


F._mask_from_url = _cached_mask


def working_image(path: Path) -> Image.Image:
    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    small, _ = detect.working_copy(original, get_settings().detect_max_edge)
    return small


def baseline_counts() -> dict[str, dict[str, int]]:
    if not BASELINE.exists():
        return {}
    rows: dict[str, dict[str, int]] = {}
    pattern = re.compile(r"^(\S.*?\.(?:jpe?g|png))\s+\d+x\d+\s+[\d.]+s\s+spans=\s*\d+\s+(.*)$")
    for line in BASELINE.read_text().splitlines():
        match = pattern.match(line)
        if match:
            rows[match.group(1)] = {
                kind: int(count) for kind, count in re.findall(r"(\w+)x(\d+)", match.group(2))
            }
    return rows


def annotate(image: Image.Image, detections, spans) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for item in detections:
        if item.kind == "roof":
            continue
        colour = detect.COLORS.get(item.kind, "#ec4899").lstrip("#")
        rgb = tuple(int(colour[i : i + 2], 16) for i in (0, 2, 4))
        tint = Image.new("RGBA", canvas.size, rgb + (110,))
        overlay.paste(tint, (0, 0), Image.fromarray(item.mask.astype(np.uint8) * 255, mode="L"))
        draw.rectangle(item.box, outline=rgb + (255,), width=2)
        draw.text((item.box[0] + 4, item.box[1] + 3), f"{item.kind} {item.score:.2f}", fill=rgb + (255,))
    for points, side, edge in spans:
        colour = (255, 193, 7) if edge == "eave" else (148, 163, 184)
        draw.line([tuple(p) for p in points], fill=colour + (255,), width=4, joint="curve")
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spend", action="store_true", help="make the uncached, billed calls")
    parser.add_argument("--photos", nargs="+", default=DEFAULT_PHOTOS)
    parser.add_argument("--kinds", nargs="+", default=None, help="subset of detect.DEFAULT_KINDS")
    parser.add_argument("--prompt", action="append", default=[], metavar="KIND=TEXT",
                        help="override one class's prompt, e.g. shrub='round shrub'")
    args = parser.parse_args()

    for override in args.prompt:
        kind, _, text = override.partition("=")
        if kind not in prompts.CLASS_QUERIES or not text:
            parser.error(f"bad --prompt {override!r}")
        prompts.CLASS_QUERIES[kind] = [text]

    selected = args.kinds or detect.DEFAULT_KINDS
    queried = prompts.query_kinds(detect.queried_kinds(selected))
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    paths = [PHOTOS / name for name in args.photos]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        parser.error(f"not found in {PHOTOS}: {missing}")

    # Plan first: exactly which prompts are uncached, before anything is spent.
    uncached = [
        (path.name, kind)
        for path in paths
        for kind in queried
        if not (CACHE / f"{cache_key(path.name, working_image(path).size, prompts.CLASS_QUERIES[kind][0], {})}.json").exists()
    ]
    print(f"{len(paths)} photos x {len(queried)} prompts; {len(uncached)} uncached "
          f"-> {len(uncached)} billed call(s), about ${len(uncached) * PRICE_PER_CALL:.2f}")
    if uncached and not args.spend:
        print("Dry run. Re-run with --spend to make those calls.")
        return 0

    baseline = baseline_counts()
    results, billed = [], 0
    for path in paths:
        image = working_image(path)
        provider = CachedFal(path.name, args.spend)
        started = time.time()
        detections = detect.run(image, provider, selected)
        billed += provider.billed

        roofs = [d.mask for d in detections if d.kind == "roof"]
        spans = roofline.dedupe(
            [s for mask in roofs for s in roofline.plane_spans(mask, image.width, True)], image.width
        ) if roofs else []
        counts: dict[str, int] = {}
        for item in detections:
            counts[item.kind] = counts.get(item.kind, 0) + 1

        annotate(image, detections, spans).save(OUT / f"{path.stem}.jpg", quality=90)
        results.append({"photo": path.name, "counts": counts, "spans": len(spans)})
        found = " ".join(f"{k}x{v}" for k, v in sorted(counts.items())) or "NOTHING"
        note = f"{time.time() - started:4.1f}s" if provider.billed else "cached"
        print(f"  {path.name:30s} {note:>7}  spans={len(spans):2d}  {found}")

    print(f"\n{'class':14s} {'fal SAM 3':>10} {'DINO+SAM 2':>11}   (photos with at least one)")
    for kind in selected:
        ours = sum(1 for r in results if r["counts"].get(kind))
        theirs = sum(1 for r in results if baseline.get(r["photo"], {}).get(kind))
        base = f"{theirs}/{len(results)}" if baseline else "n/a"
        print(f"  {kind:12s} {ours:>6}/{len(results):<3} {base:>11}")
    ours_n = np.mean([sum(v for k, v in r["counts"].items() if k in selected) for r in results])
    theirs_n = np.mean([sum(baseline.get(r["photo"], {}).values()) for r in results]) if baseline else float("nan")
    print(f"\n  regions per photo: fal {ours_n:.1f}, baseline {theirs_n:.1f}")
    print(f"  billed calls this run: {billed}")
    print(f"  overlays: {OUT}")

    (OUT / "results.json").write_text(json.dumps(
        {"prompts": {k: prompts.CLASS_QUERIES[k] for k in queried}, "photos": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
