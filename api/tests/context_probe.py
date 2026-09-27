"""Validate candidate 'sky' / 'wall' prompts for adjacency-based edge labelling.

Measurement only. Nothing in app/ is modified; this asks fal for each candidate concept
and scores how useful it would be for the two-stage edge test:

  * a roof plane's BOTTOM boundary, offset outward (downward), should land on WALL  -> eave
  * a roof plane's TOP boundary, offset outward (upward), should land on SKY        -> sky-edge

So the useful prompt is not the one with the biggest mask, it is the one that actually sits
where the test looks. Responses are cached by tests/detect_eval.py's cache, so re-runs are free.

    python tests/context_probe.py --photos "<folder>" --limit 4            # plan only
    python tests/context_probe.py --photos "<folder>" --limit 4 --spend    # make the calls
"""

import argparse
import os
import re
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))
sys.path.insert(0, str(API / "tests"))
os.environ.setdefault("SEGMENTATION_PROVIDER", "fal")

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import detect_eval as E  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.providers.prompts import SCORE_THRESHOLD  # noqa: E402
from app.services import detect, roofline  # noqa: E402

CANDIDATES = ["sky", "wall", "house", "facade"]
PRICE_PER_CALL = 0.005

# Four photos chosen to span the failure modes seen in roofline_eval:
# polygon-heavy, profile-heavy, hip roof, and a clean modern front.
DEFAULT_FOUR = [
    "Screenshot 2026-09-01 at 5.54.12 PM.png",
    "Screenshot 2026-09-01 at 5.54.28 PM.png",
    "Screenshot 2026-09-01 at 5.55.04 PM.png",
    "Screenshot 2026-09-01 at 5.55.29 PM.png",
]


def union_of(instances, shape, threshold):
    out = np.zeros(shape, dtype=bool)
    for score, mask in instances:
        if score >= threshold:
            out |= mask
    return out


def outward_hits(points: np.ndarray, target: np.ndarray, dy: int) -> float:
    """Share of boundary samples whose outward offset lands on `target`."""
    if len(points) == 0 or not target.any():
        return 0.0
    height, width = target.shape
    xs = np.clip(points[:, 0].astype(int), 0, width - 1)
    ys = np.clip(points[:, 1].astype(int) + dy, 0, height - 1)
    return float(target[ys, xs].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    folder = Path(args.photos).expanduser()
    # macOS screenshot names carry U+202F before "PM", so match on normalised whitespace
    # rather than exact bytes — otherwise this silently falls back to the wrong photos.
    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace(" ", " ").replace(" ", " ")).strip()

    available = {norm(p.name): p.name for p in folder.iterdir()
                 if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    names = [available[norm(n)] for n in DEFAULT_FOUR if norm(n) in available][: args.limit]
    if not names:
        print("WARNING: none of DEFAULT_FOUR matched; falling back to the first photos on disk")
        names = sorted(available.values())[: args.limit]

    print(f"{len(names)} photos x {len(CANDIDATES)} candidate prompts "
          f"= up to {len(names) * len(CANDIDATES)} billed calls "
          f"(~${len(names) * len(CANDIDATES) * PRICE_PER_CALL:.2f})")
    for n in names:
        print(f"  {n}")
    if not args.spend:
        print("\nPlan only. Re-run with --spend.")
        return 0

    settings = get_settings()
    billed = 0
    totals = {q: {"fires": 0, "bottom": [], "top": [], "area": []} for q in CANDIDATES}

    for name in names:
        path = folder / name
        original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        small, _ = detect.working_copy(original, settings.detect_max_edge)
        provider = E.CachedFal(path.name, True)

        roofs = [d.mask for d in provider.detect(small, ["roof"])]
        if not roofs:
            print(f"\n{name}: no roof masks, skipping")
            continue

        # Boundary sample points, in the working frame the context masks come back in.
        bottoms, tops = [], []
        for mask in roofs:
            small_mask = detect.upscale_mask(mask, small.size)
            bottoms.append(roofline.bottom_profile(small_mask))
            tops.append(roofline.top_profile(small_mask))
        bottom = np.vstack([b for b in bottoms if len(b)]) if any(len(b) for b in bottoms) else np.empty((0, 2))
        top = np.vstack([t for t in tops if len(t)]) if any(len(t) for t in tops) else np.empty((0, 2))
        offset = max(6, int(0.012 * small.width))

        print(f"\n{name}   ({len(roofs)} roof planes, offset {offset}px)")
        print(f"  {'prompt':10s} {'masks':>6s} {'best':>6s} {'area%':>7s} "
              f"{'bottom->hit':>12s} {'top->hit':>9s}")
        for query in CANDIDATES:
            before = provider.billed
            result = provider._call(small, query)
            billed += provider.billed - before
            instances = provider._instances(result, small.size)
            confident = [(s, m) for s, m in instances if s >= SCORE_THRESHOLD]
            merged = union_of(instances, (small.height, small.width), SCORE_THRESHOLD)
            area = float(merged.sum()) / (small.width * small.height) if merged.any() else 0.0
            down = outward_hits(bottom, merged, offset)
            up = outward_hits(top, merged, -offset)
            best = max((s for s, _ in instances), default=0.0)
            totals[query]["fires"] += bool(confident)
            totals[query]["bottom"].append(down)
            totals[query]["top"].append(up)
            totals[query]["area"].append(area)
            print(f"  {query:10s} {len(confident):>6d} {best:>6.2f} {100 * area:>6.1f}% "
                  f"{100 * down:>11.1f}% {100 * up:>8.1f}%")

    print("\n" + "=" * 74)
    print("SUMMARY — higher bottom->hit means a better WALL prompt (eave confirmation);")
    print("          higher top->hit means a better SKY prompt (sky-edge confirmation)")
    print("=" * 74)
    print(f"{'prompt':10s} {'fires':>7s} {'mean area':>10s} {'mean bottom->hit':>18s} {'mean top->hit':>15s}")
    for query in CANDIDATES:
        t = totals[query]
        n = max(len(t["bottom"]), 1)
        print(f"{query:10s} {t['fires']:>4d}/{len(t['bottom']):<2d} "
              f"{100 * float(np.mean(t['area'] or [0])):>9.1f}% "
              f"{100 * float(np.mean(t['bottom'] or [0])):>17.1f}% "
              f"{100 * float(np.mean(t['top'] or [0])):>14.1f}%")
    print(f"\nbilled this run: {billed}  (~${billed * PRICE_PER_CALL:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
