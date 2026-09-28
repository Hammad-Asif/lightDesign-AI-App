"""The roof-line baseline on the annotated photos. Cache only: no fal calls, no spend.

Runs the real seam — `routers.regions.compute_roof_lines`, which both /detect and the
target-house tap go through — and scores the spans against the red annotations with
tests/score.py. Writes a JSON every later step can diff against, which is what makes
"partition-only is byte-identical" checkable rather than asserted.

The masks are cached twice over: fal responses by tests/detect_eval.py in out/fal_cache, and
the assembled roof/blocker/solid arrays per photo in out/roofline-baseline/masks. After the
first run nothing touches the network, so this is free to re-run on every change.

house03 is scored but left out of the mean: its annotation paints whole roof surfaces rather
than roof lines, so nothing drawn on it can score much above 25% recall.

    EVAL_PHOTOS=<folder with photos and ground_truth_*.png> python tests/roofline_baseline.py
    ... --out out/roofline-baseline/after.json      # then diff against baseline.json
    ... --compare out/roofline-baseline/baseline.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
REPO = API.parent
sys.path.insert(0, str(API))
sys.path.insert(0, str(API / "tests"))
os.environ.setdefault("SEGMENTATION_PROVIDER", "fal")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402

import detect_eval as E  # noqa: E402  — the cached fal provider
import score as S  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.routers import regions  # noqa: E402
from app.services import detect, intent, occlusion  # noqa: E402

OUT = REPO / "out" / "roofline-baseline"          # overlays and cached masks; gitignored
MASKS = OUT / "masks"
# The baseline itself is committed, so a later step can be diffed against it on any machine.
BASELINE = API / "tests" / "baselines" / "roofline_partition.json"
PROMPT = "i want c9 bulb on roofline"
# The annotation covers roof surfaces, not roof lines, so recall is capped near 25%.
FLOODED = {"house03"}

COLOURS = {"ridge": (255, 59, 48), "hip": (255, 159, 10), "valley": (10, 132, 255),
           "eave": (57, 255, 20), "rake": (191, 90, 242), "junction": (60, 60, 60)}


def load(path: Path):
    """Roof masks, blockers, solid and scores exactly as /detect builds them."""
    MASKS.mkdir(parents=True, exist_ok=True)
    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    stored = MASKS / f"{path.stem}.npz"
    if stored.exists():
        d = np.load(stored)
        return (original, [m.astype(bool) for m in d["roofs"]],
                d["blockers"] if bool(d["has_blockers"]) else None,
                d["solid"] if bool(d["has_solid"]) else None, list(d["scores"]))
    settings = get_settings()
    edge = (settings.partition_detect_max_edge if settings.edge_extractor == "partition"
            else settings.detect_max_edge)
    small, _ = detect.working_copy(original, edge)
    provider = E.CachedFal(path.name, False, max_masks=settings.partition_fal_max_masks,
                           image_format="jpeg")
    found = detect.run(small, provider, intent.parse(PROMPT).kinds or None, settings.detect_blockers)
    for item in found:
        item.mask = detect.upscale_mask(item.mask, original.size)
    frame = (original.height, original.width)
    blockers = occlusion.blocker_union([d.mask for d in found if d.kind in occlusion.BLOCKER_KINDS], frame)
    solid = occlusion.blocker_union([d.mask for d in found if d.kind in occlusion.SOLID_KINDS], frame)
    roofs = [d for d in found if d.kind == "roof"]
    np.savez_compressed(
        stored, roofs=np.array([d.mask for d in roofs]),
        blockers=blockers if blockers is not None else np.zeros(frame, bool),
        has_blockers=blockers is not None,
        solid=solid if solid is not None else np.zeros(frame, bool), has_solid=solid is not None,
        scores=np.array([d.score for d in roofs]))
    return original, [d.mask for d in roofs], blockers, solid, [d.score for d in roofs]


def spans_for(path: Path) -> tuple[Image.Image, list]:
    """The lines /detect would draw, through the same function /detect calls."""
    original, masks, blockers, solid, scores = load(path)
    if not masks:
        return original, []
    spans, _unrequested, _details = regions.compute_roof_lines(
        masks, list(scores), blockers, solid, intent.parse(PROMPT), original,
    )
    return original, spans


def overlay(original: Image.Image, spans: list, out: Path) -> None:
    canvas = original.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for points, _side, edge in spans:
        draw.line([tuple(p) for p in points], fill=COLOURS.get(edge, (150, 150, 150)), width=5)
        for point in (points[0], points[-1]):
            draw.ellipse([point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7],
                         outline=(255, 0, 0), width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def measure() -> dict:
    """Per photo: recall, precision, F1, span and vertex counts, scored at score.TOLERANCE."""
    found = S.pairs()
    if not found:
        raise SystemExit(f"No annotated pairs under {S.PHOTOS}. Set EVAL_PHOTOS to the folder "
                         f"holding both the photos and their ground_truth_*.png.")
    settings = get_settings()
    # The baseline is only comparable against a run with the same geometry configuration,
    # so it records it rather than leaving it to whatever .env happened to say.
    out: dict = {
        "tolerance": S.TOLERANCE, "prompt": PROMPT,
        "config": {
            "edge_extractor": settings.edge_extractor,
            "partition_engine": settings.partition_engine,
            "partition_junctions": settings.partition_junctions,
            "partition_constrained_snap": settings.partition_constrained_snap,
            "partition_snap": settings.partition_snap,
            "snap_to_edges": settings.snap_to_edges,
            "interior_hips": settings.interior_hips,
            "detect_blockers": settings.detect_blockers,
        },
        "photos": {},
    }
    for stem, photo, annotation in found:
        original, spans = spans_for(photo)
        # score.py works at its own scale; spans are in full-photo pixels.
        image = S.working(photo)
        factor = image.size[0] / original.size[0]
        scaled = [([[p[0] * factor, p[1] * factor] for p in points], side, edge)
                  for points, side, edge in spans]
        targets = S.load_targets(annotation, image.size)
        recall, precision, f1, count = S.score(scaled, targets)
        out["photos"][stem] = {
            "recall": round(recall, 1), "precision": round(precision, 1), "f1": round(f1, 1),
            "spans": count, "points": sum(len(p) for p, _, _ in spans),
            "edges": {e: sum(1 for _, _, k in spans if k == e)
                      for e in sorted({k for _, _, k in spans})},
            "flooded": stem in FLOODED,
        }
        overlay(original, spans, OUT / "overlays" / f"{stem}.png")
    scored = [v for k, v in out["photos"].items() if not v["flooded"]]
    if scored:
        out["mean"] = {key: round(sum(v[key] for v in scored) / len(scored), 1)
                       for key in ("recall", "precision", "f1")}
        out["mean"]["spans"] = sum(v["spans"] for v in scored)
        out["mean"]["photos"] = len(scored)
    return out


def show(result: dict, previous: dict | None) -> bool:
    """Print the table; return True when it differs from `previous`."""
    print(f"\n{'photo':22s} {'recall':>7s} {'prec':>7s} {'F1':>7s} {'spans':>6s} {'pts':>5s}  edges")
    print("-" * 88)
    for stem, v in result["photos"].items():
        edges = " ".join(f"{k}:{n}" for k, n in v["edges"].items())
        flag = "  (not in mean: annotation floods the roof)" if v["flooded"] else ""
        print(f"{stem:22s} {v['recall']:7.1f} {v['precision']:7.1f} {v['f1']:7.1f} "
              f"{v['spans']:6d} {v['points']:5d}  {edges}{flag}")
    if "mean" in result:
        m = result["mean"]
        print("-" * 88)
        print(f"{'MEAN of ' + str(m['photos']):22s} {m['recall']:7.1f} {m['precision']:7.1f} "
              f"{m['f1']:7.1f} {m['spans']:6d}")
    if previous is None:
        return False
    if previous.get("config") and previous["config"] != result.get("config"):
        differing = {k: (v, result["config"].get(k)) for k, v in previous["config"].items()
                     if result["config"].get(k) != v}
        print(f"\nWARNING: configured differently from the baseline: {differing}")
    changed = [s for s in result["photos"]
               if previous.get("photos", {}).get(s) != result["photos"][s]]
    print()
    if changed:
        print(f"CHANGED against the baseline on {len(changed)} photo(s): {', '.join(changed)}")
        for stem in changed:
            was, now = previous.get("photos", {}).get(stem), result["photos"][stem]
            print(f"  {stem}: F1 {was and was.get('f1')} -> {now['f1']}, "
                  f"spans {was and was.get('spans')} -> {now['spans']}")
    else:
        print("IDENTICAL to the baseline on every photo.")
    return bool(changed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(BASELINE))
    parser.add_argument("--compare", nargs="?", const=str(BASELINE),
                        help="an earlier JSON to diff against (default: the committed baseline)")
    args = parser.parse_args()

    settings = get_settings()
    print(f"extractor {settings.edge_extractor}/{settings.partition_engine}  "
          f"sources {getattr(settings, 'roofline_sources', settings.roofline_detector)}  "
          f"photos {S.PHOTOS}")
    result = measure()
    previous = json.loads(Path(args.compare).read_text()) if args.compare else None
    changed = show(result, previous)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nwrote {target}\noverlays {OUT / 'overlays'}")
    return 1 if (previous is not None and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
