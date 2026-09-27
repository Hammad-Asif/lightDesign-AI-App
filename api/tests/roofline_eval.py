"""Measure the current roofline pipeline on a folder of photos. Measurement only.

Runs detection and roofline extraction exactly as POST /detect does — working copy,
detect.run, upscale to full resolution, plane_spans per plane, dedupe, blocker union,
bridge, hidden stretches — then renders an overlay per photo and writes
ground-truth-free statistics.

Nothing in roofline.py / detect.py / fal.py / prompts.py is modified. This module only
calls them and reports what they did.

Every fal response is cached by tests/detect_eval.py's cache in out/fal_cache/, keyed by
photo name, posted size, prompt, model and max_masks, so re-runs cost nothing.

    python tests/roofline_eval.py --photos "<folder>"           # plan only, no spend
    python tests/roofline_eval.py --photos "<folder>" --spend   # make the uncached calls
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

API = Path(__file__).resolve().parents[1]
REPO = API.parent
sys.path.insert(0, str(API))
sys.path.insert(0, str(API / "tests"))
os.environ.setdefault("SEGMENTATION_PROVIDER", "fal")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

import detect_eval as E  # noqa: E402  — reuses its cache, CachedFal and local-mask patch
from app.config import get_settings  # noqa: E402
from app.providers import prompts as P  # noqa: E402
from app.services import detect, intent, occlusion, roofline  # noqa: E402

OUT = REPO / "out" / "roofline-eval"
OVERLAYS = OUT / "overlays"
PRICE_PER_CALL = 0.005

# Auto-flag thresholds. Deliberately blunt: these only decide what a human looks at first.
COVERAGE_LOW = 0.35        # span length / plane outline perimeter
WIDE_BASE_FRAC = 0.25      # a plane this wide with no spans is suspicious
MAX_PLAUSIBLE_PLANES = 8   # matches MAX_PER_CLASS, which roof bypasses entirely
ONMASK_LOW = 0.5           # share of a span's samples sitting on a roof mask
SCORE_LOW = 0.5

PLANE_COLOURS = [
    (56, 189, 248), (168, 85, 247), (34, 197, 94), (249, 115, 22),
    (236, 72, 153), (250, 204, 21), (20, 184, 166), (239, 68, 68),
]
SPAN_COLOUR = (255, 214, 0)
HIDDEN_COLOUR = (10, 10, 12)

# Sibling concepts for adjacency labelling. Validated on 4 photos (tests/context_probe.py):
# "wall" fired 0/4. "sky" was then ablated out: alone it changed 1 label of 29, because slope
# already separates ridge from rake. "facade" drives the other 28, but those are unproven —
# mostly flips to eave, which a gable's wall triangle can cause wrongly — so this path exists
# for human scoring only and is NOT used by /detect.
CONTEXT_QUERIES = ["facade"]


def peaked(mask: np.ndarray) -> bool:
    """Does this mask's own silhouette contain an interior apex?

    If it does, the ridge is inside that mask and no seam between masks can ever be it, so
    the ridge is reported as unrecoverable rather than manufactured from a boundary.
    """
    profile = roofline.top_profile(mask)
    if len(profile) < 20:
        return False
    xs, ys = profile[:, 0].astype(float), profile[:, 1].astype(float)
    window = max(3, len(ys) // 25) | 1
    smooth = np.convolve(ys, np.ones(window) / window, mode="same")
    core = smooth[window:-window]
    if len(core) < 3:
        return False
    apex = int(np.argmin(core)) + window
    fraction = (xs[apex] - xs[0]) / max(xs[-1] - xs[0], 1)
    rows = np.where(mask.any(axis=1))[0]
    height = max(rows[-1] - rows[0], 1) if rows.size else 1
    rise = min(smooth[window] - smooth[apex], smooth[-window - 1] - smooth[apex])
    return bool(0.15 < fraction < 0.85 and rise > 0.05 * height)


def build_context(provider, small: Image.Image, size, roof_union: np.ndarray):
    """(sky, wall) at full resolution, or None if nothing usable came back.

    The "facade" concept overlaps the roof by ~93% of roof area (measured), so an
    unsubtracted mask would vote "wall" for every boundary and the eave test would be
    meaningless. The roof is removed before it is used as a wall proxy.
    """
    from app.providers.prompts import SCORE_THRESHOLD

    resolved = {}
    for query in CONTEXT_QUERIES:
        instances = provider._instances(provider._call(small, query), small.size)
        merged = np.zeros((small.height, small.width), dtype=bool)
        for score, mask in instances:
            if score >= SCORE_THRESHOLD:
                merged |= mask
        resolved[query] = detect.upscale_mask(merged, size) if merged.any() else None

    sky = resolved.get("sky")
    wall = resolved.get("facade")
    if wall is not None and roof_union is not None:
        wall = wall & ~roof_union
    if sky is None and wall is None:
        return None
    return sky, wall


def font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


# ---- geometry helpers (read-only use of the production modules) ---------------


def polyline_length(points) -> float:
    a = np.asarray(points, dtype=float)
    if len(a) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def plane_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    return float(cv2.arcLength(max(contours, key=cv2.contourArea), True))


def slice_polyline(points, start: float, end: float):
    """The part of a polyline between two distances from its start."""
    part = []
    walked = 0.0
    pts = np.asarray(points, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        if length == 0:
            continue
        lo, hi = max(start, walked), min(end, walked + length)
        if hi > lo:
            at = lambda d: a + (b - a) * ((d - walked) / length)  # noqa: E731
            if not part:
                part.append(at(lo))
            part.append(at(hi))
        walked += length
        if walked >= end:
            break
    return part


def on_mask_fraction(points, mask: np.ndarray) -> float:
    a = np.asarray(points, dtype=float)
    if len(a) < 2:
        return 0.0
    coords, _ = occlusion._samples(a)
    return float(occlusion._on_mask(mask, coords).mean())


def plane_report(mask: np.ndarray, image_width: int, hybrid: bool):
    """What plane_spans() produced, which path it took, and how much it discarded.

    Recomputes the same decisions by calling roofline's own helpers, so the counts
    describe the real code rather than a re-implementation of it.
    """
    profile = [
        (pts, side, edge)
        for edge in ("eave", "ridge")
        for pts, side in roofline.extract_spans(mask, edge)
    ]
    used_polygon = False
    if hybrid:
        if not profile:
            used_polygon = True
        else:
            longest = max(polyline_length(p) for p, *_ in profile)
            if longest < roofline.PROFILE_MIN_FRAC * max(roofline._plane_width(mask), 1):
                used_polygon = True

    spans = roofline.plane_spans(mask, image_width, hybrid)

    dropped = 0
    if used_polygon:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if contours:
            outline = max(contours, key=cv2.contourArea)
            corners = cv2.approxPolyDP(
                outline, roofline.POLY_EPSILON * cv2.arcLength(outline, True), True
            ).reshape(-1, 2)
            for a, b in zip(corners, np.roll(corners, -1, axis=0)):
                if float(np.linalg.norm(b.astype(float) - a)) < roofline.POLY_MIN_LEN * image_width:
                    dropped += 1
    else:
        height, width = mask.shape
        minimum = roofline.MIN_SPAN_FRACTION * width
        for edge in ("eave", "ridge"):
            prof = roofline.bottom_profile(mask) if edge == "eave" else roofline.top_profile(mask)
            if len(prof) < 2:
                continue
            for run in roofline._runs(prof, width, height):
                if len(run) < 2 or polyline_length(run) < minimum:
                    dropped += 1
                    continue
                for piece in roofline._split_run(run, width):
                    if not (len(piece) >= 2 and polyline_length(piece) >= minimum):
                        dropped += 1
    return spans, used_polygon, dropped


# ---- overlay -----------------------------------------------------------------


def dashed(draw, points, colour, width, on=18, off=12):
    pts = np.asarray(points, dtype=float)
    total = polyline_length(pts)
    pos = 0.0
    while pos < total:
        piece = slice_polyline(pts, pos, min(pos + on, total))
        if len(piece) >= 2:
            draw.line([tuple(p) for p in piece], fill=colour, width=width, joint="curve")
        pos += on + off


def render(photo: Image.Image, planes, spans, hidden, path: Path, header: str):
    canvas = photo.convert("RGBA")
    tint = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    for index, (mask, _score) in enumerate(planes):
        colour = PLANE_COLOURS[index % len(PLANE_COLOURS)]
        layer = Image.new("RGBA", canvas.size, colour + (70,))
        tint.paste(layer, (0, 0), Image.fromarray((mask.astype(np.uint8) * 255), mode="L"))
    canvas = Image.alpha_composite(canvas, tint)

    draw = ImageDraw.Draw(canvas)
    line_width = max(3, canvas.width // 500)
    label_font = font(max(15, canvas.width // 90))

    for index, (points, side, edge) in enumerate(spans):
        draw.line([tuple(p) for p in points], fill=SPAN_COLOUR + (255,), width=line_width, joint="curve")
        for start, end in hidden[index]:
            piece = slice_polyline(points, start, end)
            if len(piece) >= 2:
                dashed(draw, piece, HIDDEN_COLOUR + (255,), line_width + 2)
        mid = points[len(points) // 2]
        text = f"{index}:{side}/{edge}"
        x, y = float(mid[0]), float(mid[1]) - label_font.size - 4
        draw.text((x + 1, y + 1), text, font=label_font, fill=(0, 0, 0, 220))
        draw.text((x, y), text, font=label_font, fill=SPAN_COLOUR + (255,))

    bar_height = label_font.size * 3
    out = Image.new("RGBA", (canvas.width, canvas.height + bar_height), (18, 18, 20, 255))
    out.paste(canvas, (0, bar_height))
    head = ImageDraw.Draw(out)
    head.text((12, 8), header, font=label_font, fill=(235, 235, 235, 255))
    head.text(
        (12, 10 + label_font.size),
        "plane tint = roof mask   yellow = extracted span   dashed dark = hidden behind an object",
        font=font(max(12, canvas.width // 120)),
        fill=(170, 170, 175, 255),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(path, quality=92)


# ---- per photo ---------------------------------------------------------------


def run_photo(path: Path, spend: bool, kinds, context_on: bool = False):
    settings = get_settings()
    original = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    small, _ = detect.working_copy(original, settings.detect_max_edge)

    provider = E.CachedFal(path.name, spend)
    started = time.time()
    detections = detect.run(small, provider, list(kinds))

    # Exactly as routers/regions.py does it: full resolution before any geometry.
    for item in detections:
        item.mask = detect.upscale_mask(item.mask, original.size)

    frame = (original.height, original.width)
    blockers = occlusion.blocker_union(
        [d.mask for d in detections if d.kind in occlusion.BLOCKER_KINDS], frame
    )
    roofs = [(d.mask, float(d.score)) for d in detections if d.kind == "roof"]
    width = roofs[0][0].shape[1] if roofs else original.width
    hybrid = settings.edge_extractor == "hybrid"

    roof_union = np.zeros(frame, dtype=bool)
    for mask, _ in roofs:
        roof_union |= mask
    context = build_context(provider, small, original.size, roof_union) if (context_on and roofs) else None

    per_plane, raw, raw_before = [], [], []
    dropped_total = 0
    changed = 0
    apex_planes = 0
    for index, (mask, score) in enumerate(roofs):
        before, polygon, dropped = plane_report(mask, width, hybrid)
        # Same geometry either way — only the labels differ — so the two align one to one.
        spans = roofline.plane_spans(mask, width, hybrid, context) if context else before
        changed += sum(1 for (_, _, a), (_, _, b) in zip(before, spans) if a != b)
        apex_planes += peaked(mask)
        dropped_total += dropped
        raw_before.extend(before)
        raw.extend(spans)
        x = np.where(mask.any(axis=0))[0]
        y = np.where(mask.any(axis=1))[0]
        perimeter = plane_perimeter(mask)
        covered = sum(polyline_length(p) for p, *_ in spans)
        per_plane.append(
            {
                "plane": index,
                "score": round(score, 3),
                "width_frac": round((len(x) / width) if len(x) else 0.0, 3),
                "area_frac": round(float(mask.sum()) / (mask.shape[0] * mask.shape[1]), 4),
                "path": "polygon" if polygon else "profile",
                "spans": len(spans),
                "dropped": dropped,
                "perimeter_px": round(perimeter, 1),
                "coverage": round(covered / perimeter, 3) if perimeter else 0.0,
                "bbox": [int(x[0]), int(y[0]), int(x[-1]), int(y[-1])] if len(x) and len(y) else [0, 0, 0, 0],
            }
        )

    # A ridge only exists where two planes meet, so it comes from abutting masks, never
    # from one mask's own boundary.
    seams = roofline.seam_ridges([m for m, _ in roofs], width) if len(roofs) > 1 else []
    raw = raw + seams

    def finish(candidates):
        deduped = roofline.dedupe(candidates, width) if candidates else []
        bridged = occlusion.bridge(deduped, blockers, width) if deduped else []
        return deduped, bridged

    deduped_before, bridged_before = finish(raw_before)
    deduped, bridged = finish(raw)
    hidden = [occlusion.hidden_stretches(p, blockers, width) for p, *_ in bridged]

    roof_union = np.zeros(frame, dtype=bool)
    for mask, _ in roofs:
        roof_union |= mask
    if roof_union.any():
        roof_union = cv2.dilate(roof_union.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    span_rows = []
    for index, (points, side, edge) in enumerate(bridged):
        span_rows.append(
            {
                "span": index,
                "side": side,
                "edge": edge,
                "length_px": round(polyline_length(points), 1),
                "on_roof_mask": round(on_mask_fraction(points, roof_union), 3) if roof_union.any() else 0.0,
                "hidden_stretches": len(hidden[index]),
                "hidden_px": round(sum(e - s for s, e in hidden[index]), 1),
            }
        )

    return {
        "photo": path.name,
        "size": original.size,
        "elapsed": time.time() - started,
        "billed": provider.billed,
        "planes": roofs,
        "plane_rows": per_plane,
        "dropped_min_span": dropped_total,
        "raw_spans": len(raw),
        "deduped": len(deduped),
        "context_used": context is not None,
        "label_changes": changed,
        "apex_planes": apex_planes,
        "seam_ridges": len(seams),
        "bridged_before": bridged_before,
        "bridged": bridged,
        "hidden": hidden,
        "span_rows": span_rows,
        "image": original,
    }


def flags_for(result, calls_per_photo) -> list[str]:
    out = []
    planes = result["plane_rows"]
    if not planes:
        out.append("NO_ROOF_DETECTED")
    if len(planes) > MAX_PLAUSIBLE_PLANES:
        out.append(f"MANY_PLANES({len(planes)})")
    for plane in planes:
        if plane["spans"] == 0 and plane["width_frac"] >= WIDE_BASE_FRAC:
            out.append(f"WIDE_PLANE_NO_SPANS(p{plane['plane']})")
        if plane["spans"] and plane["coverage"] < COVERAGE_LOW:
            out.append(f"LOW_COVERAGE(p{plane['plane']}={plane['coverage']})")
        if plane["score"] < SCORE_LOW:
            out.append(f"LOW_SCORE(p{plane['plane']}={plane['score']})")
    if planes and all(p["path"] == "polygon" for p in planes):
        out.append("ALL_POLYGON_PATH_LABELS_UNRELIABLE")
    stray = [s["span"] for s in result["span_rows"] if s["on_roof_mask"] < ONMASK_LOW]
    if stray:
        out.append(f"SPANS_OFF_MASK({','.join(str(s) for s in stray)})")
    if planes and not result["span_rows"]:
        out.append("PLANES_BUT_NO_SPANS")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True, help="folder of photos to measure")
    parser.add_argument("--spend", action="store_true", help="make the uncached, billed calls")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context", action="store_true",
                        help=f"also prompt {CONTEXT_QUERIES} and label edges by adjacency "
                             f"(+{len(CONTEXT_QUERIES)} billed calls per photo)")
    parser.add_argument("--prompt", action="append", default=[],
                        help="apply /detect's instruction filters to the extracted spans and "
                             "report what survives (repeatable; no extra calls)")
    args = parser.parse_args()

    folder = Path(args.photos).expanduser()
    if not folder.is_dir():
        parser.error(f"not a folder: {folder}")
    photos = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.limit:
        photos = photos[: args.limit]
    if not photos:
        parser.error(f"no images in {folder}")

    kinds = detect.DEFAULT_KINDS
    queried = P.query_kinds(detect.queried_kinds(list(kinds)))
    if args.context:
        queried = queried + CONTEXT_QUERIES
    per_photo = len(queried)
    print(f"{len(photos)} photos x {per_photo} prompts ({', '.join(queried)})")
    print(f"worst case {len(photos) * per_photo} billed calls, about ${len(photos) * per_photo * PRICE_PER_CALL:.2f}")
    if not args.spend:
        print("\nPlan only. Re-run with --spend to make the uncached calls.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    OVERLAYS.mkdir(parents=True, exist_ok=True)

    results, billed_total = [], 0
    for path in photos:
        result = run_photo(path, args.spend, kinds, args.context)
        billed_total += result["billed"]
        result["flags"] = flags_for(result, per_photo)
        render(
            result["image"],
            result["planes"],
            result["bridged"],
            result["hidden"],
            OVERLAYS / f"{path.stem}.png",
            f"{path.name}   planes={len(result['planes'])}  spans={len(result['bridged'])}  "
            f"billed={result['billed']}",
        )
        result.pop("image")
        results.append(result)
        note = f"{result['elapsed']:5.1f}s" if result["billed"] else "cached"
        print(
            f"  {path.name[:44]:44s} {note:>7}  planes={len(result['planes']):2d} "
            f"spans={len(result['bridged']):2d}  {' '.join(result['flags'])[:60]}"
        )

    write_csvs(results, per_photo)
    print_tables(results, per_photo, billed_total)
    if args.prompt:
        prompt_report(results, args.prompt)
    return 0


def write_csvs(results, per_photo):
    stats = OUT / "stats.csv"
    with stats.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "photo", "width", "height", "planes", "mean_score", "spans", "eave", "ridge",
            "front", "left", "right", "coverage_mean", "coverage_min", "dropped_min_span",
            "dropped_dedupe", "bridges", "hidden_count", "hidden_px", "polygon_planes",
            "billed_calls", "cached_calls", "flags",
        ])
        for r in results:
            planes = r["plane_rows"]
            spans = r["span_rows"]
            cov = [p["coverage"] for p in planes if p["spans"]]
            writer.writerow([
                r["photo"], r["size"][0], r["size"][1], len(planes),
                round(float(np.mean([p["score"] for p in planes])), 3) if planes else "",
                len(spans),
                sum(1 for s in spans if s["edge"] == "eave"),
                sum(1 for s in spans if s["edge"] == "ridge"),
                sum(1 for s in spans if s["side"] == "front"),
                sum(1 for s in spans if s["side"] == "left"),
                sum(1 for s in spans if s["side"] == "right"),
                round(float(np.mean(cov)), 3) if cov else "",
                round(min(cov), 3) if cov else "",
                r["dropped_min_span"],
                r["raw_spans"] - r["deduped"],
                r["deduped"] - len(r["bridged"]),
                sum(s["hidden_stretches"] for s in spans),
                round(sum(s["hidden_px"] for s in spans), 1),
                sum(1 for p in planes if p["path"] == "polygon"),
                r["billed"], per_photo - r["billed"],
                " ".join(r["flags"]),
            ])

    with (OUT / "planes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["photo", "plane", "score", "width_frac", "area_frac", "path", "spans",
                         "dropped_min_span", "perimeter_px", "coverage", "bbox"])
        for r in results:
            for p in r["plane_rows"]:
                writer.writerow([r["photo"], p["plane"], p["score"], p["width_frac"], p["area_frac"],
                                 p["path"], p["spans"], p["dropped"], p["perimeter_px"], p["coverage"],
                                 json.dumps(p["bbox"])])

    with (OUT / "spans.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["photo", "span", "side", "edge", "length_px", "on_roof_mask",
                         "hidden_stretches", "hidden_px"])
        for r in results:
            for s in r["span_rows"]:
                writer.writerow([r["photo"], s["span"], s["side"], s["edge"], s["length_px"],
                                 s["on_roof_mask"], s["hidden_stretches"], s["hidden_px"]])

    sheet = OUT / "scoring_sheet_blank.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "photo", "planes_detected_auto", "spans_auto", "polygon_path_planes_auto",
            "roofs_present", "roofs_detected_correctly", "front_eave_fully_covered",
            "missed_lines", "false_lines", "side_label_errors", "edge_label_errors",
            "occlusion_ok", "notes",
        ])
        for r in results:
            writer.writerow([
                r["photo"], len(r["plane_rows"]), len(r["span_rows"]),
                sum(1 for p in r["plane_rows"] if p["path"] == "polygon"),
                "", "", "", "", "", "", "", "", "",
            ])


def _angle(points) -> float:
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    return float(np.degrees(np.arctan2(abs(b[1] - a[1]), max(abs(b[0] - a[0]), 1e-6))))


def _short(name: str) -> str:
    match = re.search(r"(\d+\.\d+\.\d+)", name)
    return match.group(1) if match else name[:12]


def prompt_report(results, prompts: list[str]):
    """Which spans /detect's instruction filters keep, applied exactly as routers/regions.py
    does: side first, then edge, with "unknown" edges kept for review. Measurement only."""
    dump = {}
    for prompt in prompts:
        parsed = intent.parse(prompt)
        print("\n" + "=" * 116)
        print(f'PROMPT "{prompt}"   kinds={parsed.kinds}  sides={parsed.sides or "any"}  '
              f'edges={parsed.edges or "any"}')
        print("=" * 116)
        print(f"{'photo':9s} {'spans':>5s} {'kept':>5s} | {'kept eave':>9s} {'rake':>5s} {'ridge':>6s} "
              f"{'unk':>4s} | {'drop:side':>9s} {'drop:edge':>9s} | {'kept sloped':>11s} "
              f"{'drop sloped':>11s} | {'kept off-mask':>13s}")
        print("-" * 116)
        dump[prompt] = {"sides": parsed.sides, "edges": parsed.edges, "photos": {}}
        totals = Counter()
        for r in results:
            if not r["bridged"]:
                continue
            kept = Counter()
            rows = []
            for i, (points, side, edge) in enumerate(r["bridged"]):
                side_ok = not parsed.sides or side in parsed.sides
                edge_ok = not parsed.edges or edge in parsed.edges or edge == "unknown"
                keep = side_ok and edge_ok
                sloped = 15 <= _angle(points) <= 75
                off = r["span_rows"][i]["on_roof_mask"] < ONMASK_LOW
                kept["spans"] += 1
                if keep:
                    kept["kept"] += 1
                    kept[f"kept_{edge}"] += 1
                    kept["kept_sloped"] += sloped
                    kept["kept_off"] += off
                else:
                    kept["drop_side" if not side_ok else "drop_edge"] += 1
                    kept["drop_sloped"] += sloped
                rows.append({"span": i, "side": side, "edge": edge, "angle": round(_angle(points), 1),
                             "kept": keep, "dropped_by": None if keep else ("side" if not side_ok else "edge")})
            totals.update(kept)
            dump[prompt]["photos"][_short(r["photo"])] = rows
            print(f"{_short(r['photo']):9s} {kept['spans']:>5d} {kept['kept']:>5d} | "
                  f"{kept['kept_eave']:>9d} {kept['kept_rake']:>5d} {kept['kept_ridge']:>6d} "
                  f"{kept['kept_unknown']:>4d} | {kept['drop_side']:>9d} {kept['drop_edge']:>9d} | "
                  f"{kept['kept_sloped']:>11d} {kept['drop_sloped']:>11d} | {kept['kept_off']:>13d}")
        print("-" * 116)
        print(f"{'TOTAL':9s} {totals['spans']:>5d} {totals['kept']:>5d} | {totals['kept_eave']:>9d} "
              f"{totals['kept_rake']:>5d} {totals['kept_ridge']:>6d} {totals['kept_unknown']:>4d} | "
              f"{totals['drop_side']:>9d} {totals['drop_edge']:>9d} | {totals['kept_sloped']:>11d} "
              f"{totals['drop_sloped']:>11d} | {totals['kept_off']:>13d}")
    (OUT / "prompt_selection.json").write_text(json.dumps(dump, indent=1))


def _counts(spans) -> dict:
    out = {"eave": 0, "ridge": 0, "rake": 0, "unknown": 0}
    for _points, _side, edge in spans:
        out[edge] = out.get(edge, 0) + 1
    return out


def label_table(results):
    """Before/after edge labels. 'before' is the same geometry labelled without adjacency."""
    print("\n" + "=" * 112)
    print("EDGE LABELS — BEFORE (no adjacency) vs AFTER (facade adjacency + seam ridges)")
    print("=" * 112)
    print(f"{'photo':26s} | {'eave':>9s} {'rake':>9s} {'ridge':>9s} {'unknown':>9s} "
          f"| {'changed':>7s} {'seams':>6s} {'apex':>5s}")
    print("-" * 112)
    totals = {"before": {}, "after": {}, "changed": 0, "seams": 0, "apex": 0}
    for r in results:
        before = _counts(r.get("bridged_before", []))
        after = _counts(r["bridged"])
        for key in ("eave", "rake", "ridge", "unknown"):
            totals["before"][key] = totals["before"].get(key, 0) + before.get(key, 0)
            totals["after"][key] = totals["after"].get(key, 0) + after.get(key, 0)
        totals["changed"] += r.get("label_changes", 0)
        totals["seams"] += r.get("seam_ridges", 0)
        totals["apex"] += r.get("apex_planes", 0)
        cell = lambda k: f"{before.get(k,0)}->{after.get(k,0)}"  # noqa: E731
        print(f"{r['photo'][11:37]:26s} | {cell('eave'):>9s} {cell('rake'):>9s} "
              f"{cell('ridge'):>9s} {cell('unknown'):>9s} | "
              f"{r.get('label_changes',0):>7d} {r.get('seam_ridges',0):>6d} {r.get('apex_planes',0):>5d}")
    print("-" * 112)
    cell = lambda k: f"{totals['before'].get(k,0)}->{totals['after'].get(k,0)}"  # noqa: E731
    print(f"{'TOTAL':26s} | {cell('eave'):>9s} {cell('rake'):>9s} {cell('ridge'):>9s} "
          f"{cell('unknown'):>9s} | {totals['changed']:>7d} {totals['seams']:>6d} {totals['apex']:>5d}")
    print(f"\n  segments relabelled by adjacency: {totals['changed']}")
    print(f"  ridges recovered from plane seams: {totals['seams']}")
    print(f"  planes whose ridge is interior (seam impossible, not manufactured): {totals['apex']}")


def print_tables(results, per_photo, billed_total):
    print("\n" + "=" * 118)
    print("PER-IMAGE STATS (no ground truth used)")
    print("=" * 118)
    head = (f"{'photo':42s} {'pl':>3s} {'score':>6s} {'sp':>3s} {'eav':>4s} {'rdg':>4s} "
            f"{'F/L/R':>9s} {'cov':>5s} {'ddp':>4s} {'brg':>4s} {'hid':>4s} {'poly':>5s}")
    print(head)
    print("-" * 118)
    for r in results:
        planes, spans = r["plane_rows"], r["span_rows"]
        cov = [p["coverage"] for p in planes if p["spans"]]
        flr = (f"{sum(1 for s in spans if s['side']=='front')}/"
               f"{sum(1 for s in spans if s['side']=='left')}/"
               f"{sum(1 for s in spans if s['side']=='right')}")
        print(
            f"{r['photo'][:42]:42s} {len(planes):>3d} "
            f"{(np.mean([p['score'] for p in planes]) if planes else 0):>6.2f} {len(spans):>3d} "
            f"{sum(1 for s in spans if s['edge']=='eave'):>4d} "
            f"{sum(1 for s in spans if s['edge']=='ridge'):>4d} {flr:>9s} "
            f"{(np.mean(cov) if cov else 0):>5.2f} {r['raw_spans']-r['deduped']:>4d} "
            f"{r['deduped']-len(r['bridged']):>4d} "
            f"{sum(s['hidden_stretches'] for s in spans):>4d} "
            f"{sum(1 for p in planes if p['path']=='polygon'):>5d}"
        )
    print("-" * 118)
    total_planes = sum(len(r["plane_rows"]) for r in results)
    total_spans = sum(len(r["span_rows"]) for r in results)
    print(f"{'TOTAL':42s} {total_planes:>3d} {'':>6s} {total_spans:>3d}")

    print("\n" + "=" * 118)
    print("AUTO-FLAGGED FOR REVIEW FIRST")
    print("=" * 118)
    flagged = [r for r in results if r["flags"]]
    if not flagged:
        print("  none")
    for r in flagged:
        print(f"  {r['photo'][:44]:44s} {', '.join(r['flags'])}")

    label_table(results)

    print(f"\nbilled fal calls this run: {billed_total}  (~${billed_total * PRICE_PER_CALL:.2f})")
    print(f"served from cache: {len(results) * per_photo - billed_total}")
    print(f"overlays: {OVERLAYS}")
    print(f"stats:    {OUT / 'stats.csv'}")
    print(f"sheet:    {OUT / 'scoring_sheet_blank.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
