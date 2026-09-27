"""Target-house selection (TARGET_HOUSE_ONLY): flag off against flag on, on the same detections.

Detection runs once per photo through the production path — partition mode's 2000 px working
copy, 12 instances per call, JPEG, blockers on — and the result is cached in
out/target-eval/cache. Every run after the first costs nothing, and flag off and flag on are
compared on identical SAM 3 output, so any difference between them is this feature's doing.

    python tests/target_eval.py --photos "<folder>"            # plan: which photos need detection
    python tests/target_eval.py --photos "<folder>" --spend    # detect the uncached ones (4 calls each)
    python tests/target_eval.py --photos "<folder>" --planes   # also draw every plane, numbered

Reports, per photo, the roof planes kept and dropped, the lines the flag takes away, and the
acceptance check: no roof line lost on the target house (same edge type and side, ends within
ENDPOINT_FRACTION of the width). Landscape regions come from the same detection both ways, so
they are identical here by construction; tests/smoke.py checks that through the API.

Which lines count as the target house's is judged by the planes they lie on, much as the flag
itself judges them, so a target line wrongly taken away would pass as a neighbour's. The
overlays are the independent check: the flag-on side draws every line the flag took away dashed
in magenta, and each one should sit on a red, dropped roof. Overlays go to out/target-eval/overlays.
"""

import argparse
import os
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
REPO = API.parent
sys.path.insert(0, str(API))
os.environ.setdefault("SEGMENTATION_PROVIDER", "fal")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.providers.base import Detection  # noqa: E402
from app.services import detect, intent, occlusion, target  # noqa: E402

OUT = REPO / "out" / "target-eval"
CACHE = OUT / "cache"
PLANES = OUT / "planes"
OVERLAYS = OUT / "overlays"
PROMPT = "C9 warm white on the roofline"
PROMPT_KINDS = ["roof"]          # what PROMPT asks for; blockers come with it
PRICE_PER_CALL = 0.005
CALLS_PER_PHOTO = 4              # roof, car, palm tree, tree
ENDPOINT_FRACTION = 0.01         # of the width: how far a target line's end may move with the flag on
EDGE_COLOURS = {"eave": (57, 255, 20), "rake": (191, 90, 242), "hip": (255, 159, 10), "valley": (10, 132, 255),
                "ridge": (255, 59, 48), "junction": (58, 58, 60), "unknown": (56, 189, 248)}

PLANE_COLOURS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
]


def load_photo(path: Path) -> Image.Image:
    """The photo as the app stores it: EXIF orientation applied, RGB, long edge at most MAX_IMAGE_EDGE."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    edge = get_settings().max_image_edge
    if max(image.size) > edge:
        image.thumbnail((edge, edge), Image.LANCZOS)
    return image


def _cache_path(path: Path) -> Path:
    return CACHE / f"{path.stem}.npz"


def cached(path: Path) -> bool:
    return _cache_path(path).exists()


def detections(path: Path, image: Image.Image, spend: bool) -> list[Detection] | None:
    """detect.run's output for this photo at full resolution, from the cache or from fal."""
    target = _cache_path(path)
    if target.exists():
        data = np.load(target, allow_pickle=False)
        size = tuple(int(v) for v in data["working"])
        found = []
        for index, kind in enumerate(data["kinds"]):
            small = np.unpackbits(data["masks"][index])[: size[0] * size[1]].reshape(size[1], size[0]).astype(bool)
            found.append(Detection(kind=str(kind), label=str(data["labels"][index]),
                                   mask=detect.upscale_mask(small, image.size),
                                   score=float(data["scores"][index]), box=tuple(int(v) for v in data["boxes"][index])))
        return found
    if not spend:
        return None

    from app.providers.fal import FalProvider

    settings = get_settings()
    small_image, _ = detect.working_copy(image, settings.partition_detect_max_edge)
    provider = FalProvider(max_masks=settings.partition_fal_max_masks, image_format="jpeg")
    found = detect.run(small_image, provider, PROMPT_KINDS, settings.detect_blockers)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        working=np.array(small_image.size),
        kinds=np.array([d.kind for d in found]),
        labels=np.array([d.label for d in found]),
        scores=np.array([d.score for d in found], dtype=float),
        boxes=np.array([d.box for d in found], dtype=int).reshape(-1, 4),
        masks=np.stack([np.packbits(d.mask.ravel()) for d in found]) if found else np.zeros((0, 0), np.uint8),
    )
    for d in found:
        d.mask = detect.upscale_mask(d.mask, image.size)
    return found


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_planes(image: Image.Image, found: list[Detection], out: Path, keep: list[bool] | None = None) -> None:
    """Every roof plane in its own colour with its index; kept planes outlined white, dropped red."""
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    roofs = [d for d in found if d.kind == "roof"]
    blockers = [d for d in found if d.kind != "roof"]
    for d in blockers:
        tint = Image.new("RGBA", base.size, (120, 130, 150, 70))
        layer.paste(tint, (0, 0), Image.fromarray(d.mask.astype(np.uint8) * 255))
    for index, d in enumerate(roofs):
        colour = PLANE_COLOURS[index % len(PLANE_COLOURS)]
        tint = Image.new("RGBA", base.size, colour + (110,))
        layer.paste(tint, (0, 0), Image.fromarray(d.mask.astype(np.uint8) * 255))
    composed = Image.alpha_composite(base, layer)
    draw = ImageDraw.Draw(composed)
    width = max(2, round(max(image.size) / 500))
    font = _font(max(14, round(max(image.size) / 45)))
    for index, d in enumerate(roofs):
        contours, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge = (255, 255, 255, 255) if keep is None or keep[index] else (255, 0, 0, 255)
        for contour in contours:
            if len(contour) >= 3:
                draw.line([tuple(p[0]) for p in contour] + [tuple(contour[0][0])], fill=edge, width=width)
        ys, xs = np.nonzero(d.mask)
        if len(xs):
            label = f"{index}:{d.score:.2f}"
            x, y = int(xs.mean()), int(ys.mean())
            draw.text((x, y), label, fill=(255, 255, 255, 255), font=font, stroke_width=3, stroke_fill=(0, 0, 0, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(out, quality=88)


def photos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def lines_both_ways(image: Image.Image, found: list[Detection]) -> dict:
    """The roof lines /detect draws with the flag off and with it on, from the same detections.

    Both go through routers.regions.compute_roof_lines, as /detect and the tap do, so the only
    differences are the ones the flag makes: the dropped planes' segments do not vote on the
    vanishing points, and their lines are taken away (target.on_house).
    """
    from app.routers.regions import compute_roof_lines

    frame = (image.height, image.width)
    blockers = occlusion.blocker_union([d.mask for d in found if d.kind in occlusion.BLOCKER_KINDS], frame)
    solid = occlusion.blocker_union([d.mask for d in found if d.kind in occlusion.SOLID_KINDS], frame)
    roofs = [d for d in found if d.kind == "roof"]
    masks, scores = [d.mask for d in roofs], [d.score for d in roofs]
    parsed = intent.parse(PROMPT)
    off = compute_roof_lines(masks, scores, blockers, solid, parsed, image)
    keep = target.select(masks, image)
    on = compute_roof_lines(masks, scores, blockers, solid, parsed, image, vp_region=target.evidence_region(masks, keep))
    return {"masks": masks, "keep": keep, "off": off[0] + off[1],
            "on": target.on_house(on[0], masks, keep) + target.on_house(on[1], masks, keep)}


def _share_on(mask: np.ndarray, points) -> float:
    """Share of a line's length that lies on `mask`."""
    array = np.asarray(points, float)
    samples = []
    for a, b in zip(array[:-1], array[1:]):
        steps = max(2, int(np.linalg.norm(b - a) / 2))
        samples += [a + (b - a) * t for t in np.linspace(0, 1, steps)]
    samples = np.asarray(samples)
    height, width = mask.shape
    xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, height - 1)
    return float(mask[ys, xs].mean())


def _ends_match(a, b, limit: float) -> bool:
    a0, a1, b0, b1 = (np.asarray(p, float) for p in (a[0], a[-1], b[0], b[-1]))
    same = max(np.linalg.norm(a0 - b0), np.linalg.norm(a1 - b1))
    flipped = max(np.linalg.norm(a0 - b1), np.linalg.norm(a1 - b0))
    return min(same, flipped) <= limit


def acceptance(image: Image.Image, both: dict) -> dict:
    """The client's checks: every target-house line with the flag off is still there with it on,
    with the same edge type and side and both ends within ENDPOINT_FRACTION of the width."""
    width = image.width
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(0.01 * width) + 1,) * 2)
    kept = np.zeros((image.height, width), np.uint8)
    dropped = np.zeros((image.height, width), np.uint8)
    for mask, keep_it in zip(both["masks"], both["keep"]):
        (kept if keep_it else dropped)[mask] = 1
    kept, dropped = cv2.dilate(kept, grow).astype(bool), cv2.dilate(dropped, grow).astype(bool)

    limit = ENDPOINT_FRACTION * width
    lost, changed, target_lines = [], [], 0
    for points, side, edge in both["off"]:
        if _share_on(kept, points) <= _share_on(dropped, points):
            continue
        target_lines += 1
        if any(e == edge and s == side and _ends_match(points, p, limit) for p, s, e in both["on"]):
            continue
        near = [(e, s) for p, s, e in both["on"] if _ends_match(points, p, limit)]
        (changed if near else lost).append((edge, side, near[0] if near else None))
    return {"target_lines": target_lines, "lost": lost, "changed": changed,
            "neighbour_lines_off": len(both["off"]) - target_lines}


def removed_lines(image: Image.Image, both: dict) -> list[tuple]:
    """Flag-off lines with no flag-on line within ENDPOINT_FRACTION, whatever its label: what
    the flag took away."""
    limit = ENDPOINT_FRACTION * image.width
    return [line for line in both["off"] if not any(_ends_match(line[0], p, limit) for p, _s, _e in both["on"])]


def _dashes(points: list[tuple], length: float) -> list[list[tuple]]:
    """A polyline cut into dashes `length` long, with gaps as long between them."""
    pieces: list[list[tuple]] = []
    current = [points[0]]
    drawing, left = True, length          # in a dash or a gap, and how much of it remains
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        span = float(np.linalg.norm(b - a))
        done = 0.0
        while span - done > left:
            done += left
            point = tuple(a + (b - a) * (done / span))
            if drawing:
                pieces.append(current + [point])
            current = [point]
            drawing, left = not drawing, length
        left -= span - done
        current = current + [tuple(b)] if drawing else [tuple(b)]
    if drawing and len(current) > 1:
        pieces.append(current)
    return pieces


def draw_compare(image: Image.Image, both: dict, out: Path) -> None:
    """Flag off on the left, flag on on the right; on the right, dropped planes are shaded red
    and the lines the flag took away are drawn dashed in magenta, where they were. Every line
    has a dark outline, as in the editor, so it reads against sky and shingles alike."""
    # About 3 px on screen with the sheet seen whole, two photos side by side.
    width = max(3, round(max(image.size) / 300))
    outline = width + 2 * max(1, width // 2)

    def stroke(draw: ImageDraw.ImageDraw, points, colour: tuple, dashed: bool = False) -> None:
        points = [tuple(p) for p in points]
        pieces = _dashes(points, 3 * width) if dashed else [points]
        for piece in pieces:
            draw.line(piece, fill=(0, 0, 0, 170), width=outline, joint="curve")
        for piece in pieces:
            draw.line(piece, fill=colour + (255,), width=width, joint="curve")

    def panel(lines, shade: bool) -> Image.Image:
        base = image.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        if shade:
            for mask, keep_it in zip(both["masks"], both["keep"]):
                if not keep_it:
                    layer.paste(Image.new("RGBA", base.size, (255, 40, 40, 90)), (0, 0),
                                Image.fromarray(mask.astype(np.uint8) * 255))
        draw = ImageDraw.Draw(layer)
        for points, _side, edge in lines:
            stroke(draw, points, EDGE_COLOURS.get(edge, (255, 255, 255)))
        if shade:
            for points, _side, _edge in removed_lines(image, both):
                stroke(draw, points, (255, 0, 255), dashed=True)
        return Image.alpha_composite(base, layer).convert("RGB")

    left, right = panel(both["off"], False), panel(both["on"], True)
    gap = max(8, image.width // 100)
    label_h = max(24, image.height // 18)
    sheet = Image.new("RGB", (image.width * 2 + gap, image.height + label_h), "white")
    sheet.paste(left, (0, label_h))
    sheet.paste(right, (image.width + gap, label_h))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(14, label_h * 2 // 3))
    draw.text((8, 2), "flag off", fill="black", font=font)
    draw.text((image.width + gap + 8, 2), "flag on (red = not the target house, dashed magenta = lines taken away)",
              fill="black", font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True, action="append", help="folder of photos; repeatable")
    parser.add_argument("--spend", action="store_true", help="run fal detection for uncached photos")
    parser.add_argument("--planes", action="store_true", help="draw every roof plane, numbered")
    args = parser.parse_args()

    paths = [p for folder in args.photos for p in photos_in(Path(folder).expanduser())]
    missing = [p for p in paths if not cached(p)]
    print(f"{len(paths)} photos, {len(missing)} not detected yet"
          f" (~${len(missing) * CALLS_PER_PHOTO * PRICE_PER_CALL:.2f} to detect)")
    if missing and not args.spend:
        for p in missing:
            print(f"  needs detection: {p.name}")
        print("Re-run with --spend to detect them.")

    print(f"\n{'photo':28s} {'planes':>7s} {'dropped':>9s} {'target lines':>13s} {'lost':>5s} {'changed':>8s}"
          f" {'lines off/on':>13s} {'taken away':>11s}")
    failures = 0
    for path in paths:
        image = load_photo(path)
        found = detections(path, image, args.spend)
        if found is None:
            continue
        if args.planes:
            draw_planes(image, found, PLANES / f"{path.stem}.jpg")
        both = lines_both_ways(image, found)
        result = acceptance(image, both)
        draw_compare(image, both, OVERLAYS / f"{path.stem}.jpg")
        dropped = [index for index, keep_it in enumerate(both["keep"]) if not keep_it]
        print(f"{path.name[:28]:28s} {len(both['keep']):>7d} {str(dropped):>9s} {result['target_lines']:>13d}"
              f" {len(result['lost']):>5d} {len(result['changed']):>8d} {len(both['off']):>6d}/{len(both['on']):<6d}"
              f" {len(removed_lines(image, both)):>11d}")
        for edge, side, _ in result["lost"]:
            print(f"    lost: {edge} ({side})")
        for edge, side, now in result["changed"]:
            print(f"    changed: {edge} ({side}) -> {now[0]} ({now[1]})")
        failures += bool(result["lost"] or result["changed"])
    print(f"\n{failures} photo(s) with a target line lost or changed. Overlays: {OVERLAYS}\n"
          "Check there that every dashed magenta line (taken away) lies on a red roof (not the target house).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
