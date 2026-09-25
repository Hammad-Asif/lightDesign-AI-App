"""Evaluation harness. Nothing gets tuned before this can score it.

  python -m eval.harness                          phased pipeline, every photo, default params
  python -m eval.harness --pipeline legacy        the pre-rebuild pipeline, as the baseline to beat
  python -m eval.harness --split heldout          only the held-out photos
  python -m eval.harness --params params.json --name tuned

A photo takes part when eval/photos/<stem>.* has an annotation (eval/annotations/<stem>.json)
and cached SAM instances (eval/cache/<stem>.npz), so runs are free and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from eval import metrics
from roofline.config import Params
from roofline.watermark import is_watermarked

ROOT = Path(__file__).resolve().parent
RED, BOLD, DIM, RESET = "\033[91m", "\033[1m", "\033[2m", "\033[0m"


@dataclass
class Item:
    stem: str
    image_path: Path
    annotation: dict
    cache_path: Path
    ignore_path: Optional[Path]

    def image(self) -> np.ndarray:
        return cv2.imread(str(self.image_path))

    def instances(self) -> list[dict]:
        data = np.load(self.cache_path, allow_pickle=False)
        scores = [None if np.isnan(s) else float(s) for s in data["scores"]]
        return [{"mask": m, "score": s, "source": "cache"} for m, s in zip(data["masks"], scores)]

    def ignore(self) -> Optional[np.ndarray]:
        return cv2.imread(str(self.ignore_path), cv2.IMREAD_GRAYSCALE) if self.ignore_path else None


def discover() -> list[Item]:
    items = []
    for image_path in sorted((ROOT / "photos").glob("*")):
        if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        stem = image_path.stem
        ann, cache, ignore = ROOT / "annotations" / f"{stem}.json", ROOT / "cache" / f"{stem}.npz", ROOT / "ignore" / f"{stem}.png"
        if not ann.exists() or not cache.exists():
            print(f"{DIM}skipping {stem}: needs {'an annotation' if not ann.exists() else 'cached SAM instances'}{RESET}")
            continue
        items.append(Item(stem, image_path, json.loads(ann.read_text()), cache, ignore if ignore.exists() else None))
    return items


def load_split(items: list[Item]) -> dict[str, list[str]]:
    """About a third of the photos are held out, chosen by a hash of the name so it never drifts."""
    path = ROOT / "split.json"
    split = json.loads(path.read_text()) if path.exists() else {"train": [], "heldout": []}
    known = set(split["train"]) | set(split["heldout"])
    for item in items:
        if item.stem not in known:
            bucket = "heldout" if int(hashlib.sha1(item.stem.encode()).hexdigest(), 16) % 3 == 0 else "train"
            split[bucket].append(item.stem)
    path.write_text(json.dumps(split, indent=1))
    return split


def watermark_list() -> set[str]:
    path = ROOT / "watermarked.txt"
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def check_watermark(item: Item, image: np.ndarray) -> bool:
    flagged, reason = is_watermarked(image, item.stem, watermark_list())
    if flagged:
        print(f"{RED}{BOLD}!!! WATERMARK WARNING: {item.stem} is watermarked ({reason}). "
              f"Watermarks pollute line detection. Replace it with the clean original. !!!{RESET}")
    return flagged


# ---- pipelines -----------------------------------------------------------------------------

def run_legacy(image, instances, params: Params, ignore) -> list:
    """The pipeline as it stood before the rebuild: per-section polygons straight from the masks."""
    from eval import legacy_app as L
    cleaned = [i for i in (L.make_instance(x["mask"], "cache", x["score"]) for x in instances) if i]
    contours = [(i, c) for i, inst in enumerate(cleaned) for c in L.usable_contours(inst["mask"])]
    if not contours:
        return []
    result = L.extract_roofline(contours, params.epsilon, 0.0, refine=True)
    return [[*s["points"][0], *s["points"][1]] for s in result["segments"]]


def run_phased(image, instances, params: Params, ignore, stop_after: int = 7) -> list:
    from roofline import pipeline
    result = pipeline.run(image, instances, params, ignore_mask=ignore, stop_after=stop_after)
    return [[*s["points"][0], *s["points"][1]] for s in result["segments"]]


PIPELINES: dict[str, Callable] = {"legacy": run_legacy, "phased": run_phased}


def evaluate_items(items: list[Item], predictor: Callable, params: Params, quiet: bool = False, **kwargs) -> dict:
    rows = {}
    for item in items:
        image = item.image()
        if not quiet:
            check_watermark(item, image)
        started = time.perf_counter()
        segments = predictor(image, item.instances(), params, item.ignore(), **kwargs)
        elapsed = (time.perf_counter() - started) * 1000
        rows[item.stem] = {**metrics.evaluate(segments, item.annotation, image.shape[:2]), "runtime_ms": round(elapsed, 1)}
        rows[item.stem]["score"] = round(metrics.score(rows[item.stem]), 4)
    return rows


def summarise(rows: dict) -> dict:
    if not rows:
        return {}
    keys = [f"f1@{f * 100:g}%" for f in metrics.TOLERANCES_FRAC]
    angular = [r[keys[1]]["mean_angular_error_deg"] for r in rows.values() if r[keys[1]]["mean_angular_error_deg"] is not None]
    return {"photos": len(rows),
            "chamfer_pct": round(float(np.mean([r["chamfer_pct"] for r in rows.values()])), 3),
            "chamfer_px": round(float(np.mean([r["chamfer_px"] for r in rows.values()])), 2),
            **{k: round(float(np.mean([r[k]["f1"] for r in rows.values()])), 3) for k in keys},
            "angular_error_deg": round(float(np.mean(angular)), 2) if angular else None,
            "unmatched_predictions": int(sum(r[keys[2]]["unmatched_predictions"] for r in rows.values())),
            "n_pred": int(sum(r["n_pred"] for r in rows.values())), "n_gt": int(sum(r["n_gt"] for r in rows.values())),
            "score": round(float(np.mean([r["score"] for r in rows.values()])), 4),
            "runtime_ms": round(float(np.mean([r["runtime_ms"] for r in rows.values()])), 1)}


def print_table(rows: dict, title: str) -> None:
    keys = [f"f1@{f * 100:g}%" for f in metrics.TOLERANCES_FRAC]
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'photo':30s} {'chamfer':>9s} {'F1@0.5':>7s} {'F1@1':>6s} {'F1@2':>6s} {'ang°':>6s} {'pred/gt':>8s} {'unmatched':>9s} {'ms':>7s}")
    for stem, r in rows.items():
        ang = r[keys[1]]["mean_angular_error_deg"]
        print(f"{stem:30s} {r['chamfer_pct']:8.3f}% {r[keys[0]]['f1']:7.3f} {r[keys[1]]['f1']:6.3f} {r[keys[2]]['f1']:6.3f} "
              f"{(f'{ang:.2f}' if ang is not None else '-'):>6s} {r['n_pred']:>4d}/{r['n_gt']:<3d} {r[keys[2]]['unmatched_predictions']:>9d} {r['runtime_ms']:7.1f}")
    s = summarise(rows)
    if s:
        print(f"{BOLD}{'mean':30s} {s['chamfer_pct']:8.3f}% {s[keys[0]]:7.3f} {s[keys[1]]:6.3f} {s[keys[2]]:6.3f} "
              f"{(str(s['angular_error_deg']) if s['angular_error_deg'] is not None else '-'):>6s} {s['n_pred']:>4d}/{s['n_gt']:<3d} {s['unmatched_predictions']:>9d} {s['runtime_ms']:7.1f}   score {s['score']}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="phased")
    parser.add_argument("--split", choices=["all", "train", "heldout"], default="all")
    parser.add_argument("--params", type=Path, help="JSON file of Params overrides (default: params.json if present)")
    parser.add_argument("--stop-after", type=int, default=7, help="phased pipeline: stop after this phase (3..7)")
    parser.add_argument("--name", default=None, help="save results to eval/results/<name>.json")
    args = parser.parse_args()

    items = discover()
    split = load_split(items)
    if args.split != "all":
        items = [i for i in items if i.stem in split[args.split]]
    params = Params.load(args.params) if args.params else Params.load()
    kwargs = {"stop_after": args.stop_after} if args.pipeline == "phased" else {}
    rows = evaluate_items(items, PIPELINES[args.pipeline], params, **kwargs)
    print_table(rows, f"{args.pipeline} pipeline, split={args.split}" + (f", stop after phase {args.stop_after}" if kwargs else ""))
    if args.name:
        out = ROOT / "results" / f"{args.name}.json"
        out.write_text(json.dumps({"pipeline": args.pipeline, "split": args.split, "params": params.to_dict(), "summary": summarise(rows), "photos": rows}, indent=1))
        print(f"saved {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    sys.exit(main())
