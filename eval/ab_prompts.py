"""A/B test of SAM 3 prompting on a real photo, scored with the harness metrics.

  python -m eval.ab_prompts owner_house_annotated

Each variant's instances come from the live API once and are cached on disk (.sam_cache), so
repeating the comparison is free. Variants:
  roof                      the baseline text prompt
  roof plane                does a more specific concept give cleaner per-plane instances?
  roof - X                  the baseline with "dormer" / "chimney" / "skylight" cut out of the planes
  roof + tree pass          foliage from a SAM "tree" pass instead of the colour heuristic
  box + text                does a box prompt work together with non-empty text? (points do not)
"""

from __future__ import annotations

import sys

import numpy as np

from eval import harness, metrics
from roofline import pipeline, sam
from roofline.config import Params


def evaluate(item, image, instances, params, foliage=None) -> dict:
    prep = pipeline.prepare(image, instances, params, ignore_mask=item.ignore(), foliage_override=foliage)
    result = pipeline.solve(prep, params)
    segs = [[*s["points"][0], *s["points"][1]] for s in result["segments"]]
    row = metrics.evaluate(segs, item.annotation, image.shape[:2])
    row["score"], row["planes"] = metrics.score(row), len(result["planes"])
    return row


def main() -> None:
    stem = sys.argv[1] if len(sys.argv) > 1 else "owner_house_annotated"
    item = {i.stem: i for i in harness.discover()}[stem]
    image, params = item.image(), Params.load()
    if not sam.configured():
        sys.exit("SAM 3 is not configured (FAL_KEY).")
    rows = {}
    roof = sam.text_instances(image, "roof")
    rows["roof"] = (len(roof), evaluate(item, image, roof, params))
    plane = sam.text_instances(image, "roof plane")
    rows["roof plane"] = (len(plane), evaluate(item, image, plane, params))
    for concept in ("dormer", "chimney", "skylight"):
        found = sam.text_instances(image, concept)
        cut = np.zeros(image.shape[:2], bool)
        for f in found:
            cut |= f["mask"] > 0
        carved = [{**i, "mask": np.where(cut, 0, i["mask"]).astype(np.uint8)} for i in roof]
        rows[f"roof - {concept} ({len(found)} found)"] = (len(carved), evaluate(item, image, carved, params))
    trees = sam.concept_union(image, ["tree"])
    rows[f"roof + tree pass ({trees.mean() / 255 * 100:.0f}% of image)"] = (len(roof), evaluate(item, image, roof, params, foliage=trees))

    biggest = max(roof, key=lambda i: int((i["mask"] > 0).sum()))
    ys, xs = np.nonzero(biggest["mask"])
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    boxed = sam.box_instances(image, "roof", box)
    same = len(boxed) == len(roof) and all(np.array_equal(a["mask"], b["mask"]) for a, b in zip(boxed, roof))
    inside = [float(((i["mask"] > 0)[box[1]:box[3], box[0]:box[2]]).sum()) / max(float((i["mask"] > 0).sum()), 1) for i in boxed]
    print(f"box + text: {len(boxed)} instances; identical to text-only: {same}; share of each instance inside the box: {[round(v, 2) for v in inside]}")
    print("  -> " + ("the box is IGNORED when text is given (same as points)" if same else "the box changes the result, so it is honoured together with text"))

    print(f"\n{'variant':42s} {'inst':>4s} {'planes':>6s} {'chamfer':>9s} {'F1@0.5':>7s} {'F1@1':>6s} {'F1@2':>6s} {'unmatched':>9s} {'score':>7s}")
    for name, (n, r) in rows.items():
        print(f"{name:42s} {n:4d} {r['planes']:6d} {r['chamfer_pct']:8.3f}% {r['f1@0.5%']['f1']:7.3f} {r['f1@1%']['f1']:6.3f} {r['f1@2%']['f1']:6.3f} {r['f1@2%']['unmatched_predictions']:9d} {r['score']:7.4f}")


if __name__ == "__main__":
    main()
