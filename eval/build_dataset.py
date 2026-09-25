"""Write the evaluation set to eval/photos, eval/annotations and eval/cache.

  python -m eval.build_dataset            synthetic scenes (exact ground truth, SAM-like masks)
  python -m eval.build_dataset --owner P  also register the owner's annotated photo P, with the
                                          ground truth traced from their blue strokes and the
                                          SAM 3 masks cached from a real call

Annotation format (one JSON per photo): closed polygons, one per roof plane.
  {"image": "x.png", "width": W, "height": H,
   "planes": [{"id": 0, "polygon": [[x, y], ...], "skip_edges": [k, ...]}],
   "lines": [[[x1, y1], [x2, y2]], ...]}            # optional strokes that close no polygon
SAM cache (one .npz per photo): masks (N, H, W) uint8 0/255, scores (N,), prompt.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from roofline import synth

ROOT = Path(__file__).resolve().parent
WORKING_MAX_DIM = 2000


def save_cache(stem: str, masks: list[np.ndarray], scores: list[float], prompt: str, **extra: np.ndarray) -> None:
    """`extra`: optional obstacle masks (occluder_solid, occluder_leafy), as a SAM obstacle pass would give them."""
    np.savez_compressed(ROOT / "cache" / f"{stem}.npz", masks=np.stack(masks).astype(np.uint8),
                        scores=np.array([np.nan if s is None else s for s in scores], np.float32), prompt=prompt,
                        **{k: v.astype(np.uint8) for k, v in extra.items()})


def build_synthetic(specs=None, seed_offset: int = 0) -> None:
    for i, spec in enumerate(specs if specs is not None else synth.SCENE_SPECS):
        scene, instances = synth.make_scene(spec, seed=seed_offset + i)
        cv2.imwrite(str(ROOT / "photos" / f"{scene.name}.png"), scene.image)
        h, w = scene.image.shape[:2]
        annotation = {"image": f"{scene.name}.png", "width": w, "height": h, "source": "synthetic, exact",
                      "camera": {"f_px": scene.camera.f, "pitch_deg": round(scene.camera.pitch_deg, 3), "yaw_deg": round(scene.camera.yaw_deg, 3)},
                      "shared_edges": scene.shared_edges,
                      "planes": [{"id": p["id"], "name": p["name"], "polygon": p["polygon"], "true_pitch_deg": p["true_pitch_deg"]} for p in scene.planes]}
        (ROOT / "annotations" / f"{scene.name}.json").write_text(json.dumps(annotation, indent=1))
        occ = synth.sam_like_occluders(scene, seed_offset + i)   # the obstacle pass, SAM-like: wobbled outlines
        extra = {"occluder_solid": occ["solid"], "occluder_leafy": occ["leafy"]} if occ else {}
        save_cache(scene.name, [x["mask"] for x in instances], [x["score"] for x in instances], "synthetic", **extra)
        print(f"  {scene.name}: {len(scene.planes)} planes, {len(instances)} instances")


# The owner's hand annotation of their test photo, traced from the blue strokes they drew on it
# (working-image pixels, long edge 2000). Edges listed in skip_edges were not drawn by the owner
# (hidden behind a tree or a palm); `lines` are strokes that do not close a polygon.
OWNER_ANNOTATION = {
    "planes": [
        {"id": 0, "name": "main upper hip", "polygon": [[474, 414], [687, 332], [705, 344], [856, 283], [1108, 367], [1138, 366], [990, 449], [936, 451], [938, 425], [976, 423], [976, 393], [850, 340], [717, 390], [710, 430], [512, 437]]},
        {"id": 1, "name": "upper right hip", "polygon": [[990, 449], [1138, 366], [1152, 355], [1409, 443], [1394, 455]]},
        {"id": 2, "name": "left low roof", "polygon": [[427, 463], [743, 603], [403, 599]], "skip_edges": [2]},
        {"id": 3, "name": "garage roof", "polygon": [[1020, 657], [1445, 455], [1516, 480], [1335, 569], [1407, 630], [1307, 628], [1119, 646]], "skip_edges": [3, 5]},
        {"id": 4, "name": "front gable, left rake band", "polygon": [[1335, 569], [1516, 480], [1553, 486], [1390, 562]]},
        {"id": 5, "name": "far right roof", "polygon": [[1775, 577], [1868, 623], [1774, 631]], "skip_edges": [2]},
    ],
    "lines": [[[1550, 487], [1701, 557]], [[774, 378], [929, 381]], [[434, 466], [500, 447]], [[1848, 696], [1924, 692]]],
}


def build_owner(photo: Path, probe_dir: Path) -> None:
    stem = "owner_house_annotated"
    image = cv2.imread(str(photo))
    h0, w0 = image.shape[:2]
    scale = min(1.0, WORKING_MAX_DIM / max(h0, w0))
    if scale < 1:
        image = cv2.resize(image, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    cv2.imwrite(str(ROOT / "photos" / f"{stem}.png"), image)
    annotation = {"image": f"{stem}.png", "width": w, "height": h,
                  "source": "traced from the owner's hand-drawn blue strokes; approximate to about 10 px; review before trusting small deltas",
                  **OWNER_ANNOTATION}
    (ROOT / "annotations" / f"{stem}.json").write_text(json.dumps(annotation, indent=1))

    # the strokes are not part of the scene: keep image-based line detection away from them
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    strokes = cv2.inRange(hsv, (98, 140, 170), (112, 255, 255))
    strokes = cv2.dilate(cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), np.ones((9, 9), np.uint8))
    cv2.imwrite(str(ROOT / "ignore" / f"{stem}.png"), strokes)

    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in sorted(glob.glob(str(probe_dir / "probe_text_mask*.png")))]
    scores = json.loads((probe_dir / "probe_text.json").read_text())["scores"]
    masks = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) if m.shape != (h, w) else m for m in masks]
    save_cache(stem, masks, scores, "roof")
    print(f"  {stem}: {len(OWNER_ANNOTATION['planes'])} planes, {len(masks)} real SAM 3 instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", type=Path, help="the owner's annotated photo")
    parser.add_argument("--obstacles-only", action="store_true", help="only (re)build the sample houses with obstacles in front")
    parser.add_argument("--probe-dir", type=Path, help="folder holding probe_text_mask*.png and probe_text.json from a real SAM 3 call")
    args = parser.parse_args()
    for sub in ("photos", "annotations", "cache", "ignore", "results"):
        (ROOT / sub).mkdir(exist_ok=True)
    if not args.obstacles_only:
        print("synthetic scenes:")
        build_synthetic()
    print("sample houses with obstacles:")
    build_synthetic(synth.OBSTACLE_SPECS, seed_offset=100)
    if args.owner:
        print("owner photo:")
        build_owner(args.owner, args.probe_dir)
