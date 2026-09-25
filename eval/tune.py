"""Phase 8: the outer tuning loop. The inner loop (roofline.graph) optimises each image; this loop
optimises the pipeline's thresholds across images, and every change it proposes comes with metric
deltas on photos it did not see.

  python -m eval.tune                 random search + coordinate refinement on the train split
  python -m eval.tune --trials 120    more search
  python -m eval.tune --write         save params.json if the held-out score does not get worse

Geometry-only parameters are searched against cached `prepare` results, so a trial costs
milliseconds per photo. Parameters that change the partition or the trust flags force a re-prepare
and get a smaller budget.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from eval import harness, metrics
from roofline import pipeline
from roofline.config import PARAMS_FILE, Params

SOLVE_SPACE = {            # name: (low, high, log-scale?)
    "epsilon": (0.004, 0.02, True), "merge_angle_deg": (3.0, 9.0, False), "snap_radius_frac": (0.004, 0.016, True),
    "line_offset_frac": (0.003, 0.012, True), "w_junc": (5.0, 120.0, True), "w_dir": (0.5, 30.0, True),
    "vp_snap_deg": (2.0, 7.0, False), "huber_inlier_px": (1.5, 4.0, False), "min_trusted_frac": (0.2, 0.6, False),
    "corner_guard_frac": (0.01, 0.04, True), "crease_min_support": (0.45, 0.8, False),
}
PREPARE_SPACE = {
    "gap_close_frac": (0.003, 0.012, True), "close_frac": (0.004, 0.014, True), "straight_rms_px": (2.0, 4.5, False),
    "min_solid_fraction": (0.05, 0.2, False), "score_floor": (0.25, 0.5, False),
}


def sample(space: dict, rng) -> dict:
    out = {}
    for name, (lo, hi, log) in space.items():
        out[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi))
    return out


class Evaluator:
    def __init__(self, items):
        self.items = items
        self.images = {i.stem: i.image() for i in items}
        self.instances = {i.stem: i.instances() for i in items}
        self.prepared: dict[tuple, dict] = {}

    def _prepare(self, params: Params) -> dict:
        key = tuple(round(getattr(params, k), 6) for k in PREPARE_SPACE)
        if key not in self.prepared:
            if len(self.prepared) > 6:
                self.prepared.pop(next(iter(self.prepared)))
            self.prepared[key] = {i.stem: pipeline.prepare(self.images[i.stem], self.instances[i.stem], params, i.ignore()) for i in self.items}
        return self.prepared[key]

    def rows(self, params: Params, stems: list[str]) -> dict:
        prepared = self._prepare(params)
        out = {}
        for item in self.items:
            if item.stem not in stems:
                continue
            result = pipeline.solve(prepared[item.stem], params)
            segs = [[*s["points"][0], *s["points"][1]] for s in result["segments"]]
            row = metrics.evaluate(segs, item.annotation, self.images[item.stem].shape[:2])
            row["score"], row["runtime_ms"] = metrics.score(row), result["timings"]["interactive_ms"]
            out[item.stem] = row
        return out

    def score(self, params: Params, stems: list[str]) -> float:
        return float(np.mean([r["score"] for r in self.rows(params, stems).values()]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="save params.json when held-out does not regress")
    args = parser.parse_args()

    items = harness.discover()
    split = harness.load_split(items)
    train, heldout = split["train"], split["heldout"]
    print(f"train: {len(train)} photos, held out: {len(heldout)} photos")
    ev = Evaluator(items)
    rng = np.random.default_rng(args.seed)
    base = Params()
    best, best_score = base, ev.score(base, train)
    print(f"default params: train score {best_score:.4f}")
    started = time.time()

    for t in range(args.trials):                                   # 1. random search, geometry parameters
        candidate = replace(best if t % 3 else base, **sample(SOLVE_SPACE, rng))
        s = ev.score(candidate, train)
        if s < best_score - 1e-6:
            best, best_score = candidate, s
            print(f"  trial {t:3d}: train score {s:.4f}")
    for sweep in range(2):                                         # 2. coordinate refinement
        for name, (lo, hi, log) in SOLVE_SPACE.items():
            current = getattr(best, name)
            for factor in (0.8, 0.9, 1.1, 1.25):
                value = float(np.clip(current * factor, lo, hi))
                candidate = replace(best, **{name: value})
                s = ev.score(candidate, train)
                if s < best_score - 1e-6:
                    best, best_score = candidate, s
                    print(f"  refine {name} -> {value:.4g}: train score {s:.4f}")
    for t in range(max(8, args.trials // 5)):                      # 3. partition / trust parameters (each needs a re-prepare)
        candidate = replace(best, **sample(PREPARE_SPACE, rng))
        s = ev.score(candidate, train)
        if s < best_score - 1e-6:
            best, best_score = candidate, s
            print(f"  prepare trial {t:3d}: train score {s:.4f}")
    print(f"search took {time.time() - started:.0f} s")

    report = {}
    for name, stems in (("train", train), ("heldout", heldout)):
        before, after = harness.summarise(ev.rows(base, stems)), harness.summarise(ev.rows(best, stems))
        report[name] = {"default": before, "tuned": after}
        print(f"\n{name} ({len(stems)} photos)            default -> tuned")
        for key in ("score", "chamfer_pct", "f1@0.5%", "f1@1%", "f1@2%", "angular_error_deg", "unmatched_predictions"):
            if before.get(key) is not None and after.get(key) is not None:
                print(f"  {key:24s} {before[key]:>9} -> {after[key]:<9}  ({after[key] - before[key]:+.4g})")
    changed = {k: (round(getattr(base, k), 5), round(getattr(best, k), 5)) for k in {**SOLVE_SPACE, **PREPARE_SPACE} if abs(getattr(best, k) - getattr(base, k)) > 1e-9}
    print("\nchanged parameters (default -> tuned):")
    for k, (a, b) in changed.items():
        print(f"  {k:22s} {a} -> {b}")
    out = Path(__file__).resolve().parent / "results" / "tuning.json"
    out.write_text(json.dumps({"report": report, "changed": changed, "tuned": best.to_dict()}, indent=1))
    held_ok = report["heldout"]["tuned"].get("score", 0) <= report["heldout"]["default"].get("score", 0) * 1.01 if heldout else False
    if args.write and held_ok:
        best.save(PARAMS_FILE)
        print(f"\nheld-out score did not get worse: wrote {PARAMS_FILE.name}")
    elif args.write:
        print("\nheld-out score got worse: params.json NOT written (the gain on train is overfitting)")
    print("note: with this few photos the held-out estimate is noisy; add annotated photos before trusting small deltas")


if __name__ == "__main__":
    main()
