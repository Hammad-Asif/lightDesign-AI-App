"""Measure the roofline fixes one at a time on the eval photos. Cache only: no fal calls.

Runs /detect's real roof path (routers.regions.roof_lines) for one instruction, with each fix
switched on alone against the committed baseline:

    A  top-edge lines that run into a wall with roof above become "junction"
    B  an outline side that mostly misses its plane is traced along the outline instead
    C  minimum lengths measured against the plane's width (staged, off in production)
    D  a low edge sloping under EAVE_MAX_DEG is an eave, not unknown
    E  pieces of one plane's edge that meet end to end are joined

With every fix off the output must equal out/roofline-fixes/baseline_6db42e5.json exactly,
which proves the switches isolate the fixes. Review sheets for the human check are written to
out/roofline-fixes/review/<fix>/, and judgements recorded in out/roofline-fixes/labels.json are
folded into the report.

    python tests/roofline_fixes_eval.py --photos "<folder>"
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

API = Path(__file__).resolve().parents[1]
REPO = API.parent
sys.path.insert(0, str(API))
sys.path.insert(0, str(API / "tests"))
os.environ.setdefault("SEGMENTATION_PROVIDER", "fal")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

import detect_eval as E  # noqa: E402  — cache, CachedFal, cached-mask patch
from app.config import get_settings  # noqa: E402
from app.providers.prompts import SCORE_THRESHOLD  # noqa: E402
from app.routers import regions  # noqa: E402
from app.services import detect, intent, occlusion, roofline  # noqa: E402

OUT = REPO / "out" / "roofline-fixes"
MASKS = OUT / "masks"
REVIEW = OUT / "review"
LABELS = OUT / "labels.json"
BASELINE = OUT / "baseline_6db42e5.json"
PROMPT = "i want c9 bulb on roofline"      # the customer photo's stored instruction
ONMASK_LOW = 0.5                           # same off-mask test as tests/roofline_eval.py

REAL_MARK_JUNCTIONS = roofline.mark_junctions
REAL_JOIN_PIECES = roofline.join_pieces
DEFAULTS = {"POLY_ON_MASK_MIN": roofline.POLY_ON_MASK_MIN, "EAVE_MAX_DEG": roofline.EAVE_MAX_DEG}


def configure(fixes: str) -> None:
    """Switch fixes on by letter; every other fix restores the baseline behaviour exactly."""
    roofline.POLY_ON_MASK_MIN = DEFAULTS["POLY_ON_MASK_MIN"] if "B" in fixes else 0.0
    roofline.EAVE_MAX_DEG = DEFAULTS["EAVE_MAX_DEG"] if "D" in fixes else 0.0
    roofline.PLANE_MIN_SPAN = "C" in fixes
    roofline.mark_junctions = REAL_MARK_JUNCTIONS if "A" in fixes else (lambda planes, masks, width: planes)
    roofline.join_pieces = REAL_JOIN_PIECES if "E" in fixes else (lambda spans, planes: list(spans))


def short(name: str) -> str:
    match = re.search(r"(\d+\.\d+\.\d+)", name)
    return match.group(1) if match else name[:12]


def angle(points) -> float:
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    return float(np.degrees(np.arctan2(abs(b[1] - a[1]), max(abs(b[0] - a[0]), 1e-6))))


def length(points) -> float:
    return roofline._length(np.asarray(points, float))


def key(span) -> str:
    return json.dumps([span[0], span[1], span[2]])


# ---- inputs ------------------------------------------------------------------


def load(path: Path):
    """Photo, roof masks and blocker union exactly as /detect builds them, from the cache."""
    settings = get_settings()
    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    frame = (original.height, original.width)
    stored = MASKS / f"{short(path.name)}.npz"
    if stored.exists():
        data = np.load(stored)
        roofs = list(data["roofs"])
        blockers = data["blockers"] if bool(data["has_blockers"]) else None
        return original, roofs, blockers

    small, _ = detect.working_copy(original, settings.detect_max_edge)
    detections = detect.run(small, E.CachedFal(path.name, False), intent.parse(PROMPT).kinds or None)
    for item in detections:
        item.mask = detect.upscale_mask(item.mask, original.size)
    blockers = occlusion.blocker_union(
        [d.mask for d in detections if d.kind in occlusion.BLOCKER_KINDS], frame
    )
    return original, [d.mask for d in detections if d.kind == "roof"], blockers


def reference(path: Path, original: Image.Image, roof_union: np.ndarray):
    """Cached sky and facade masks, used only to point the human check at likely cases."""
    small, _ = detect.working_copy(original, get_settings().detect_max_edge)
    provider = E.CachedFal(path.name, False)
    found = {}
    for query in ("sky", "facade"):
        entry = E.CACHE / f"{E.cache_key(path.name, small.size, query, {})}.json"
        if not entry.exists():
            found[query] = None
            continue
        merged = np.zeros((small.height, small.width), bool)
        for score, mask in provider._instances(provider._call(small, query), small.size):
            if score >= SCORE_THRESHOLD:
                merged |= mask
        found[query] = detect.upscale_mask(merged, original.size) if merged.any() else None
    wall = found["facade"] & ~roof_union if found["facade"] is not None else None
    return found["sky"], wall


def above_share(points, index, masks, sky, wall, image_width):
    """What lies just above the plane's top edge along a line: sky, wall, roof or other."""
    group = roofline.same_plane(index, masks)
    plane = np.logical_or.reduce([masks[k] for k in group])
    union = np.logical_or.reduce(masks)
    height, width = plane.shape
    tops, filled = plane.argmax(axis=0), plane.any(axis=0)
    snap = roofline.JUNCTION_SNAP_FRACTION * image_width
    reach = max(6, int(0.012 * image_width))
    votes, on_top, total = Counter(), 0, 0
    for x, y in np.rint(roofline._along(points)).astype(int):
        total += 1
        if not (0 <= x < width and 0 <= y < height) or not filled[x] or abs(int(tops[x]) - y) > snap:
            continue
        on_top += 1
        rows = np.arange(max(0, int(tops[x]) - reach), max(0, int(tops[x]) - 3))
        if rows.size == 0:
            continue
        if union[rows, x].mean() >= 0.5:
            votes["roof"] += 1
        elif sky is not None and sky[rows, x].mean() >= 0.5:
            votes["sky"] += 1
        elif wall is not None and wall[rows, x].mean() >= 0.5:
            votes["wall"] += 1
        else:
            votes["other"] += 1
    counted = max(sum(votes.values()), 1)
    return on_top / max(total, 1), {k: votes[k] / counted for k in ("sky", "wall", "roof", "other")}


# ---- per-fix detail ------------------------------------------------------------


def per_plane(masks, image_width, fixes: str):
    configure(fixes)
    return [roofline.plane_spans(mask, image_width, True) for mask in masks]


def off_mask(lines, masks) -> list[int]:
    union = cv2.dilate(np.logical_or.reduce(masks).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    flagged = []
    for index, (points, *_rest) in enumerate(lines):
        coords, _ = occlusion._samples(np.asarray(points, float))
        if float(occlusion._on_mask(union, coords).mean()) < ONMASK_LOW:
            flagged.append(index)
    return flagged


def on_own(points, mask) -> float:
    near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return roofline._on_share(near, points)


def deviation(points) -> float:
    """Largest distance of a traced line from the straight chord between its ends."""
    array = np.asarray(points, float)
    a, b = array[0], array[-1]
    direction = (b - a) / max(float(np.linalg.norm(b - a)), 1e-6)
    normal = np.array([-direction[1], direction[0]])
    return float(np.max(np.abs((array - a) @ normal)))


def covered(line, others, tolerance) -> float:
    """Share of a line's samples lying within `tolerance` of any of `others`."""
    points = roofline._along(line[0])
    hit = np.zeros(len(points), bool)
    for other in others:
        samples = roofline._along(other[0])
        hit |= np.min(np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2), axis=1) < tolerance
    return float(hit.mean())


def pieces_of(joined, candidates):
    """Which input spans a joined line was built from (a contiguous run of its points)."""
    out = []
    flat = joined[0]
    for span in candidates:
        pts = span[0]
        for seq in (pts, pts[::-1]):
            n = len(seq)
            if any(flat[i:i + n] == seq for i in range(len(flat) - n + 1)):
                out.append(span)
                break
    return out


# ---- review sheets -------------------------------------------------------------


def font(size: int):
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def crop_tile(original, masks, plane_index, lines, context_lines, caption, tile=620, margin=160):
    everything = [p for line in lines for p in line[0]]
    xs = [p[0] for p in everything]
    ys = [p[1] for p in everything]
    x0, y0 = max(0, min(xs) - margin), max(0, min(ys) - margin)
    x1, y1 = min(original.width, max(xs) + margin), min(original.height, max(ys) + margin)
    crop = original.crop((x0, y0, x1, y1)).convert("RGBA")
    scale = tile / max(crop.width, crop.height)
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    draw = ImageDraw.Draw(crop)
    shift = lambda pts: [((x - x0) * scale, (y - y0) * scale) for x, y in pts]  # noqa: E731
    if plane_index is not None:
        contours, _ = cv2.findContours(masks[plane_index].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            pts = shift(contour.reshape(-1, 2).tolist())
            if len(pts) > 1:
                draw.line(pts + [pts[0]], fill=(56, 189, 248, 200), width=1)
    for line in context_lines:
        draw.line(shift(line[0]), fill=(255, 255, 255, 170), width=2)
    for number, line in enumerate(lines):
        colour = (255, 40, 200, 255) if number == 0 else (255, 214, 0, 255)
        draw.line(shift(line[0]), fill=colour, width=4)
        for x, y in shift([line[0][0], line[0][-1]]):
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=colour, width=2)
    canvas = Image.new("RGB", (tile, tile + 34), (24, 24, 26))
    canvas.paste(crop.convert("RGB"), (0, 34))
    ImageDraw.Draw(canvas).text((6, 6), caption, font=font(18), fill=(235, 235, 235))
    return canvas


def sheets(tiles, folder: Path, per_sheet=6, columns=2):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("sheet_*.png"):
        old.unlink()
    written = []
    for start in range(0, len(tiles), per_sheet):
        group = tiles[start:start + per_sheet]
        width = max(t.width for t in group)
        height = max(t.height for t in group)
        rows = (len(group) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * width + (columns - 1) * 8, rows * height + (rows - 1) * 8), (0, 0, 0))
        for i, t in enumerate(group):
            sheet.paste(t, ((i % columns) * (width + 8), (i // columns) * (height + 8)))
        path = folder / f"sheet_{start // per_sheet + 1:02d}.png"
        sheet.save(path)
        written.append(path)
    return written


# ---- main ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    folder = Path(args.photos).expanduser()
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    parsed = intent.parse(PROMPT)
    baseline = json.loads(BASELINE.read_text())["photos"] if BASELINE.exists() else {}
    labels = json.loads(LABELS.read_text())["labels"] if LABELS.exists() else {}
    variants = ["", "A", "B", "C", "D", "E", "ABDE"]
    billed_before = len(list(E.CACHE.glob("*.json")))

    data = {}
    for path in photos:
        name = short(path.name)
        original, masks, blockers = load(path)
        entry = {"path": path, "original": original, "masks": masks, "blockers": blockers, "lines": {}}
        if masks:
            width = masks[0].shape[1]
            for variant in variants:
                configure(variant)
                entry["lines"][variant] = regions.roof_lines(masks, blockers, parsed, True)
            configure("")
        data[name] = entry

    # ---- parity -------------------------------------------------------------------------------
    print("=" * 100)
    print(f'PARITY  every fix off vs the committed baseline (instruction "{PROMPT}")')
    print("=" * 100)
    mismatched = [n for n, e in data.items()
                  if [list(s) for s in e["lines"].get("", [])] != baseline.get(n, {}).get("spans", [])]
    print(f"photos {len(data)}  with a roof {sum(1 for e in data.values() if e['masks'])}  "
          f"baseline lines {sum(len(e['lines'].get('', [])) for e in data.values())}  "
          f"identical to baseline: {'yes' if not mismatched and baseline else 'NO ' + str(mismatched)}")

    # ---- B ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIX B  off-mask lines in the output (under 50% on any roof mask, 4px tolerance)")
    print("=" * 100)
    print(f"{'photo':9s} {'lines':>5s} {'off before':>10s} {'off after':>9s} | replaced sides (plane: chord on-plane -> traced, "
          f"deviation, label before->after)")
    b_tiles, b_rows, b_total = [], [], Counter()
    for name, e in data.items():
        if not e["masks"]:
            continue
        masks, width = e["masks"], e["masks"][0].shape[1]
        before, after = e["lines"][""], e["lines"]["B"]
        off_b, off_a = off_mask(before, masks), off_mask(after, masks)
        b_total.update({"before": len(off_b), "after": len(off_a)})
        plane_b, plane_a = per_plane(masks, width, ""), per_plane(masks, width, "B")
        notes, untouched = [], 0
        for p, (spans_b, spans_a) in enumerate(zip(plane_b, plane_a)):
            assert len(spans_b) == len(spans_a), "B must not change which sides exist"
            for sb, sa in zip(spans_b, spans_a):
                if sb[0] == sa[0]:
                    untouched += 1
                    continue
                row = {"photo": name, "plane": p, "chord": sb, "traced": sa,
                       "chord_on": on_own(sb[0], masks[p]), "traced_on": on_own(sa[0], masks[p]),
                       "chord_len": length(sb[0]), "traced_len": length(sa[0]), "dev": deviation(sa[0]),
                       "in_output_before": key(sb) in {key(s) for s in before},
                       "in_output_after": key(sa) in {key(s) for s in after}}
                b_rows.append(row)
                notes.append(f"P{p}: {row['chord_on']:.0%}->{row['traced_on']:.0%} dev {row['dev']:.0f}px "
                             f"{sb[2]}->{sa[2]}")
                b_tiles.append(crop_tile(e["original"], masks, p, [sa, sb], [],
                                         f"B {name} P{p}  magenta=traced  yellow=old chord  {sb[2]}->{sa[2]}"))
        b_total["untouched"] += untouched
        print(f"{name:9s} {len(before):>5d} {len(off_b):>10d} {len(off_a):>9d} | {'; '.join(notes) or '-'}")
    print(f"{'TOTAL':9s} {'':>5s} {b_total['before']:>10d} {b_total['after']:>9d} | "
          f"{len(b_rows)} sides replaced, {b_total['untouched']} extracted sides byte-identical")

    # ---- A ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIX A  every non-eave plane line: junction share (geometry) and what the cached masks see above it")
    print("=" * 100)
    print(f"{'id':>9s} {'photo':9s} {'pl':>3s} {'label':>7s} {'ang':>4s} {'len':>5s} {'top':>4s} | {'share':>5s} {'call':>8s} "
          f"| {'sky':>4s} {'wall':>4s} {'roof':>4s} {'oth':>4s} | {'lit before':>10s} | human")
    a_rows, a_tiles = [], []
    for name, e in data.items():
        if not e["masks"]:
            continue
        masks, width = e["masks"], e["masks"][0].shape[1]
        sky, wall = reference(e["path"], e["original"], np.logical_or.reduce(masks))
        lit = {key(s) for s in e["lines"][""]}
        for p, spans in enumerate(per_plane(masks, width, "")):
            for s in spans:
                if s[2] == "eave":
                    continue
                share = roofline.junction_share(s[0], p, masks, width)
                on_top, seen = above_share(s[0], p, masks, sky, wall, width)
                ident = f"A{len(a_rows):03d}"
                row = {"id": ident, "photo": name, "plane": p, "span": s, "share": share,
                       "junction": share >= roofline.JUNCTION_MIN_SHARE, "top": on_top, "seen": seen,
                       "lit": key(s) in lit, "human": labels.get(ident)}
                a_rows.append(row)
                if on_top >= 0.5 or row["junction"]:
                    a_tiles.append((row, crop_tile(
                        e["original"], masks, p, [s], [],
                        f"{ident} {name} P{p} {s[2]} share {share:.0%}  sky {seen['sky']:.0%} wall {seen['wall']:.0%}")))
                print(f"{ident:>9s} {name:9s} {p:>3d} {s[2]:>7s} {angle(s[0]):>4.0f} {length(s[0]):>5.0f} {on_top:>4.0%} | "
                      f"{share:>5.0%} {'JUNCTION' if row['junction'] else '-':>8s} | {seen['sky']:>4.0%} {seen['wall']:>4.0%} "
                      f"{seen['roof']:>4.0%} {seen['other']:>4.0%} | {'lit' if row['lit'] else '':>10s} | {row['human'] or ''}")

    # The gate. "Wall" is what the human check saw above the line; a call on anything else is a
    # real hip, rake or ridge wrongly caught.
    walls = {"wall-butt", "wall-behind", "wall-missed"}
    reviewed = [r for r in a_rows if r["human"]]
    print(f"\nFIX A GATE, per plane (human-reviewed top-edge lines: {len(reviewed)} of {len(a_rows)} non-eave lines)")
    print(f"{'photo':9s} {'pl':>3s} {'reviewed':>8s} {'wall above':>10s} {'caught ok':>9s} {'WRONGLY CAUGHT':>14s} "
          f"{'missed':>6s} {'lit removed':>11s}  wrongly caught")
    per = defaultdict(list)
    for r in reviewed:
        per[(r["photo"], r["plane"])].append(r)
    gate = Counter()
    for (name, plane), rows in per.items():
        ok = [r for r in rows if r["junction"] and r["human"] in walls]
        bad = [r for r in rows if r["junction"] and r["human"] not in walls]
        missed = [r for r in rows if not r["junction"] and r["human"] in walls]
        wall = [r for r in rows if r["human"] in walls]
        lit = [r for r in ok + bad if r["lit"]]
        gate.update({"reviewed": len(rows), "wall": len(wall), "ok": len(ok), "bad": len(bad),
                     "missed": len(missed), "lit": len(lit), "missed_lit": sum(r["lit"] for r in missed)})
        if wall or ok or bad:
            print(f"{name:9s} {plane:>3d} {len(rows):>8d} {len(wall):>10d} {len(ok):>9d} {len(bad):>14d} {len(missed):>6d} "
                  f"{len(lit):>11d}  {' '.join(r['id'] + ':' + r['human'] for r in bad)}")
    print(f"{'TOTAL':9s} {'':>3s} {gate['reviewed']:>8d} {gate['wall']:>10d} {gate['ok']:>9d} {gate['bad']:>14d} "
          f"{gate['missed']:>6d} {gate['lit']:>11d}")
    print(f"  planes with a top-edge line reviewed and no wall above: "
          f"{sum(1 for rows in per.values() if not any(r['human'] in walls for r in rows))} (0 caught on any of them)"
          if not any(r["junction"] and r["human"] not in walls for r in reviewed) else "")
    kinds = Counter(r["human"] for r in reviewed if r["junction"])
    print(f"  what the {gate['ok'] + gate['bad']} calls were: {dict(kinds)}")
    print(f"  missed wall-above lines: {gate['missed']} (lit before: {gate['missed_lit']})")
    print(f"  GATE (real hips/rakes/ridges wrongly caught): {gate['bad']} -> {'PASS' if gate['bad'] == 0 else 'FAIL'}")

    # ---- D ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"FIX D  lines whose label changes (low edge under {DEFAULTS['EAVE_MAX_DEG']:.0f} deg -> eave)")
    print("=" * 100)
    d_rows, d_tiles = [], []
    for name, e in data.items():
        if not e["masks"]:
            continue
        masks, width = e["masks"], e["masks"][0].shape[1]
        output_before = {key(s) for s in e["lines"][""]}
        for p, (sb, sd) in enumerate(zip(per_plane(masks, width, ""), per_plane(masks, width, "D"))):
            for x, y in zip(sb, sd):
                if x[2] != y[2]:
                    ident = f"D{len(d_rows):03d}"
                    d_rows.append({"id": ident, "photo": name, "plane": p, "span": x, "new": y[2],
                                   "in_output": key(x) in output_before, "human": labels.get(ident)})
                    d_tiles.append(crop_tile(e["original"], masks, p, [x], [],
                                             f"{ident} {name} P{p} {x[2]}->{y[2]} {angle(x[0]):.0f}deg"))
                    print(f"  {ident} {name} P{p}  {x[2]}->{y[2]}  {angle(x[0]):4.1f} deg  len {length(x[0]):4.0f}  "
                          f"{'in output' if key(x) in output_before else 'deduped away'}  human: {labels.get(ident) or ''}")
    print(f"  changed: {len(d_rows)}  (reaching the output: {sum(r['in_output'] for r in d_rows)})")
    for verdict in ("eave-ok", "eave-wrong"):
        rows = [r for r in d_rows if r["human"] == verdict]
        print(f"  human {verdict}: {len(rows)} (in output {sum(r['in_output'] for r in rows)})  "
              f"{' '.join(r['id'] for r in rows)}")

    # ---- E ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIX E  panel entries per photo, and every join (plane, label, pieces)")
    print("=" * 100)
    e_total = Counter()
    for name, e in data.items():
        if not e["masks"]:
            continue
        masks, width = e["masks"], e["masks"][0].shape[1]
        before, after = e["lines"][""], e["lines"]["E"]
        e_total.update({"before": len(before), "after": len(after)})
        configure("")
        planes = [roofline.plane_spans(m, width, True) for m in masks]
        spans = roofline.dedupe([s for sp in planes for s in sp] + roofline.seam_ridges(masks, width), width)
        joined = REAL_JOIN_PIECES(spans, planes)
        fresh = [s for s in joined if not any(s is t for t in spans)]
        notes = []
        for line in fresh:
            parts = pieces_of(line, [s for s in spans if not any(s is t for t in joined)])
            owners = {p for p, sp in enumerate(planes) for part in parts if any(part is t for t in sp)}
            kinds = {(part[1], part[2]) for part in parts}
            ok = len(owners) == 1 and len(kinds) == 1
            e_total["joins"] += 1
            e_total["bad"] += not ok
            notes.append(f"P{'/'.join(map(str, sorted(owners)))} {line[2]}x{len(parts)}{'' if ok else ' CROSSED'}")
        print(f"{name:9s} entries {len(before):>2d} -> {len(after):>2d}   {'; '.join(notes) or '-'}")
    print(f"TOTAL     entries {e_total['before']} -> {e_total['after']}   joins {e_total['joins']}, "
          f"across planes or labels: {e_total['bad']}")
    configure("")

    # ---- C ------------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("FIX C (staged)  lines the plane-relative minimum adds to / removes from the output")
    print("=" * 100)
    c_rows, c_tiles, c_total = [], [], Counter()
    for name, e in data.items():
        if not e["masks"]:
            continue
        masks, width = e["masks"], e["masks"][0].shape[1]
        before, after = e["lines"][""], e["lines"]["C"]
        kb, ka = {key(s) for s in before}, {key(s) for s in after}
        added = [s for s in after if key(s) not in kb]
        removed = [s for s in before if key(s) not in ka]
        tolerance = max(4.0, 0.02 * width)
        near = lambda a, b: min(  # noqa: E731
            float(np.mean(np.linalg.norm(roofline._resample(a[0]) - roofline._resample(b[0]), axis=1))),
            float(np.mean(np.linalg.norm(roofline._resample(a[0]) - roofline._resample(b[0][::-1]), axis=1))))
        tally = Counter()
        for s in added:
            ident = f"C{len(c_rows):03d}"
            owner = max(range(len(masks)), key=lambda i: on_own(s[0], masks[i]))
            # The same line coming back with another side or label is not a recovered edge.
            replaces = next((r for r in removed if near(s, r) < tolerance), None)
            row = {"id": ident, "photo": name, "span": s, "plane": owner, "human": labels.get(ident),
                   "replaces": replaces}
            c_rows.append(row)
            tally["replaced" if replaces else (row["human"] or "unreviewed")] += 1
            c_tiles.append(crop_tile(e["original"], masks, owner, [s], before,
                                     f"{ident} {name} new {s[2]} {length(s[0]):.0f}px  white=baseline lines"))
        # A removed line is lost only if no output line still runs along it.
        lost = [r for r in removed if covered(r, after, tolerance) < 0.5]
        tally["lost"] = len(lost)
        c_total.update(tally)
        print(f"{name:9s} lines {len(before):>2d} -> {len(after):>2d} | added {len(added):>2d}: real {tally['real']:>2d}  "
              f"false {tally['false']:>2d}  unclear {tally['unclear']}  same line relabelled {tally['replaced']} | "
              f"baseline lines lost {len(lost)}")
        for r in [r for r in c_rows if r["photo"] == name and r["replaces"]]:
            old = r["replaces"]
            print(f"          {r['id']} replaces {old[2]}/{old[1]} {length(old[0]):.0f}px -> {r['span'][2]}/{r['span'][1]}")
        for s in lost:
            print(f"          lost: {s[2]} {s[1]} len {length(s[0]):.0f} {s[0][:3]}")
    print(f"TOTAL     added {len(c_rows)}: real {c_total['real']}, false {c_total['false']}, unclear {c_total['unclear']}, "
          f"same line relabelled {c_total['replaced']}; baseline lines lost {c_total['lost']}")

    # ---- combined -----------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("A+B+D+E together (the commit candidates), per photo")
    print("=" * 100)
    for name, e in data.items():
        if not e["masks"]:
            continue
        before, after = e["lines"][""], e["lines"]["ABDE"]
        cb, ca = Counter(s[2] for s in before), Counter(s[2] for s in after)
        print(f"{name:9s} lines {len(before):>2d} -> {len(after):>2d}   before {dict(cb)}   after {dict(ca)}   "
              f"off-mask {len(off_mask(before, e['masks']))} -> {len(off_mask(after, e['masks']))}")

    billed_after = len(list(E.CACHE.glob("*.json")))
    print(f"\nfal cache entries {billed_before} -> {billed_after} (any growth would mean a billed call)")

    if not args.no_sheets:
        sheets(b_tiles, REVIEW / "B")
        sheets([t for _, t in a_tiles], REVIEW / "A", per_sheet=6)
        sheets(d_tiles, REVIEW / "D")
        sheets(c_tiles, REVIEW / "C")
        (OUT / "review_index.json").write_text(json.dumps({
            "A": [r["id"] for r, _ in a_tiles], "D": [r["id"] for r in d_rows], "C": [r["id"] for r in c_rows]}))
        print(f"review sheets: {REVIEW}")
    configure("ABDE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
