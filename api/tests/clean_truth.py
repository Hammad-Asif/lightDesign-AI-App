"""Removes non-roof decoration from the ground-truth annotations.

The Gemini night renders these annotations come from light more than the roofline: garland
on balcony railings, uplit shrubs, and the glow of lit windows all survive into the red
mask. Scoring against those punishes the detector for correctly ignoring them, and worse,
rewards any change that starts drawing lines over planting.

Two filters, both deliberately blunt:

  * anything whose topmost pixel sits more than BELOW_ROOF_MARGIN below the lowest roof
    plane the detector found, and
  * anything narrower than MIN_RUN_WIDTH that is also not sitting on a roof plane.

The second filter is deliberately not a plain width cut. Palm fronds break a real eave
into several short pieces, and on the palm-heavy houses a bare width test deletes most of
the roofline. A short run resting on a roof plane is occlusion damage and is kept; only a
short run floating clear of every plane is treated as noise.

This is a one-off with human review, not part of scoring: it writes cleaned annotations
that the harness then reads directly, so the scorer never depends on the detector it is
scoring. Re-run it only when the annotations change, and look at the overlays it writes.

    python tests/clean_truth.py            # report and write overlays only
    python tests/clean_truth.py --write    # also write the cleaned annotations
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SEGMENTATION_PROVIDER", "local")

import score as S  # noqa: E402

# ---- thresholds -------------------------------------------------------------
BELOW_ROOF_MARGIN = 30      # px below the lowest roof plane before a run is called decoration
MIN_RUN_WIDTH = 40          # px; a run narrower than this AND clear of every plane is noise
ON_PLANE_MARGIN = 28        # px a plane is grown by before asking whether a run sits on it
ON_PLANE_SHARE = 0.50       # fraction of a run that must sit on a plane to count as on-roof
# -----------------------------------------------------------------------------

CLEAN_DIR = S.PHOTOS / "truth_clean"
OVERLAY_DIR = Path.home() / "Desktop" / "truth-cleaning"


def original(annotation: Path) -> Path:
    """The untouched annotation. score.pairs() prefers cleaned files once they exist, and
    cleaning an already-cleaned file would erode the truth a little more on every run."""
    return S.PHOTOS / annotation.name


def red_mask(annotation: Path) -> np.ndarray:
    a = np.asarray(Image.open(annotation).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 120) & (r - g > 60) & (r - b > 60)


def roof_bottom(masks: list[np.ndarray]) -> int:
    """The lowest pixel of any detected roof plane, in annotation coordinates."""
    return max((int(np.nonzero(m)[0].max()) for m in masks if m.any()), default=0)


def judge(mask: np.ndarray, masks: list[np.ndarray], floor: int,
          scale: float) -> tuple[np.ndarray, list]:
    """Split the annotation into kept and removed runs. scale maps annotation px to plane px."""
    kernel = np.ones((ON_PLANE_MARGIN * 2 + 1,) * 2, np.uint8)
    grown = [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(mask)
    verdicts = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < S.MIN_STROKE:
            continue
        top = stats[i, cv2.CC_STAT_TOP] * scale
        width = stats[i, cv2.CC_STAT_WIDTH] * scale
        gap = top - floor
        piece = labels == i
        ys, xs = np.nonzero(piece)
        yi = np.clip((ys * scale).astype(int), 0, grown[0].shape[0] - 1) if grown else ys
        xi = np.clip((xs * scale).astype(int), 0, grown[0].shape[1] - 1) if grown else xs
        on_roof = max((float(g[yi, xi].mean()) for g in grown), default=0.0) >= ON_PLANE_SHARE
        if gap > BELOW_ROOF_MARGIN:
            reason = f"below the roof by {gap:.0f}px"
        elif width < MIN_RUN_WIDTH and not on_roof:
            reason = f"stray fragment {width:.0f}px wide"
        else:
            reason = None
            keep |= piece
        verdicts.append((piece, reason, gap, width))
    return keep, verdicts


def planes_for(name: str) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Roof planes from the cache the experiments built, or from the detector if absent."""
    cached = Path("/tmp/planes") / f"{name.replace(' ', '_').replace('.jpg', '')}.npz"
    if cached.exists():
        d = np.load(cached)
        n, h, w = d["shape"]
        bits = np.unpackbits(d["masks"])[: n * h * w].reshape(n, h, w).astype(bool)
        return list(bits), (h, w)
    from app.providers.base import get_provider
    from app.services import detect

    photo = dict((stem, p) for stem, p, _ in S.pairs())[name]
    image = S.working(photo)
    small, _ = detect.working_copy(image, 1280)
    found = get_provider().detect(small, ["roof"])
    masks = [detect.upscale_mask(d.mask, image.size) for d in found if d.kind == "roof"]
    return [m for m in masks if m.any()], (image.size[1], image.size[0])


def overlay(photo: Path, masks: list[np.ndarray], verdicts: list, floor: int,
            scale: float, name: str) -> Path:
    image = np.asarray(S.working(photo).convert("RGB")).copy()
    union = np.zeros(image.shape[:2], bool)
    for m in masks:
        union |= m
    image[union] = (0.78 * image[union] + 0.22 * np.array([60, 110, 220])).astype(np.uint8)
    cv2.line(image, (0, floor), (image.shape[1], floor), (255, 255, 0), 2)

    kept = removed = 0
    for piece, reason, gap, width in verdicts:
        ys, xs = np.nonzero(piece)
        pts = np.stack([xs * scale, ys * scale], 1).astype(int)
        colour = (255, 40, 40) if reason else (0, 230, 0)
        kept += reason is None
        removed += reason is not None
        for x, y in pts:
            cv2.circle(image, (x, y), 2, colour, -1)
        if reason:
            cv2.putText(image, reason, (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2)
            cv2.putText(image, reason, (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 60, 60), 1)

    bar = np.full((112, image.shape[1], 3), 18, np.uint8)
    image = np.vstack([bar, image])
    F = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"{name} - ground-truth cleaning   {kept} kept, {removed} removed",
                (16, 32), F, 0.62, (255, 255, 255), 2)
    for i, (c, t) in enumerate([((0, 230, 0), "KEPT - scored against"),
                                ((255, 40, 40), "REMOVED - not a roof edge"),
                                ((255, 255, 0), "lowest detected roof plane"),
                                ((60, 110, 220), "SAM 3 roof planes")]):
        x = 16 + (i % 2) * 430
        y = 58 + (i // 2) * 24
        cv2.line(image, (x, y - 4), (x + 30, y - 4), c, 5)
        cv2.putText(image, t, (x + 40, y), F, 0.46, (228, 228, 228), 1)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = OVERLAY_DIR / f"{name.replace(' ', '_').replace('.jpg', '')}_cleaning.png"
    Image.fromarray(image).save(out)
    return out


def main() -> None:
    write = "--write" in sys.argv
    print(f"BELOW_ROOF_MARGIN={BELOW_ROOF_MARGIN}px   MIN_RUN_WIDTH={MIN_RUN_WIDTH}px\n")
    print(f"{'house':14s} {'runs':>5s} {'kept':>5s} {'removed':>8s}   reasons")
    totals = [0, 0, 0]
    for name, photo, annotation in S.pairs():
        annotation = original(annotation)
        masks, shape = planes_for(name)
        source = np.asarray(Image.open(annotation).convert("RGB"))
        scale = shape[1] / source.shape[1]
        mask = red_mask(annotation)
        keep, verdicts = judge(mask, masks, roof_bottom(masks), scale)
        removed = [v for v in verdicts if v[1]]
        print(f"{name[:14]:14s} {len(verdicts):5d} {len(verdicts)-len(removed):5d} "
              f"{len(removed):8d}   {'; '.join(r for _, r, _, _ in removed) or '-'}")
        totals = [totals[0] + len(verdicts), totals[1] + len(verdicts) - len(removed),
                  totals[2] + len(removed)]
        overlay(photo, masks, verdicts, roof_bottom(masks), scale, name)
        if write:
            CLEAN_DIR.mkdir(parents=True, exist_ok=True)
            canvas = np.asarray(Image.open(photo).convert("RGB").resize(
                (source.shape[1], source.shape[0]), Image.LANCZOS)).copy()
            canvas[keep] = (0.15 * canvas[keep] + 0.85 * np.array([255, 30, 30])).astype(np.uint8)
            Image.fromarray(canvas).save(CLEAN_DIR / annotation.name)
    print("-" * 62)
    print(f"{'TOTAL':14s} {totals[0]:5d} {totals[1]:5d} {totals[2]:8d}")
    print(f"\noverlays -> {OVERLAY_DIR}")
    print(f"cleaned annotations {'-> ' + str(CLEAN_DIR) if write else 'NOT written (pass --write)'}")


if __name__ == "__main__":
    main()
