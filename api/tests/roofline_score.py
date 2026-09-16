"""Score the filled-in human review sheet from roofline_eval.py.

roofline_eval.py writes scoring_sheet_blank.csv with the automatic columns already
filled. Copy it to scoring_sheet.csv, fill the human columns by looking at the
overlays, then run this:

    python tests/roofline_score.py                       # reads out/roofline-eval/scoring_sheet.csv
    python tests/roofline_score.py --sheet <path.csv>

Rows with any required human column left blank are skipped and reported, so a
partially filled sheet still scores.

Human columns:
  roofs_present               real roof planes on the TARGET house
  roofs_detected_correctly    of those, how many SAM 3 got
  front_eave_fully_covered    y/n
  missed_lines                real eaves/rakes the extraction failed to draw
  false_lines                 drawn spans not on a real roof edge
  side_label_errors           front/left/right wrong
  edge_label_errors           eave/ridge wrong
  occlusion_ok                y/n
"""

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SHEET = REPO / "out" / "roofline-eval" / "scoring_sheet.csv"

COUNTS = [
    "roofs_present", "roofs_detected_correctly", "missed_lines", "false_lines",
    "side_label_errors", "edge_label_errors",
]
YESNO = ["front_eave_fully_covered", "occlusion_ok"]


def number(value):
    value = (value or "").strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def yesno(value):
    value = (value or "").strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", default=str(DEFAULT_SHEET))
    args = parser.parse_args()

    sheet = Path(args.sheet).expanduser()
    if not sheet.exists():
        print(f"No sheet at {sheet}\nFill scoring_sheet_blank.csv and save it as scoring_sheet.csv.")
        return 2

    rows = list(csv.DictReader(sheet.open()))
    scored, skipped = [], []
    for row in rows:
        values = {name: number(row.get(name)) for name in COUNTS}
        values.update({name: yesno(row.get(name)) for name in YESNO})
        if any(value is None for value in values.values()):
            skipped.append(row.get("photo", "?"))
            continue
        values["photo"] = row.get("photo", "?")
        values["planes_detected_auto"] = number(row.get("planes_detected_auto")) or 0
        values["spans_auto"] = number(row.get("spans_auto")) or 0
        values["polygon_path_planes_auto"] = number(row.get("polygon_path_planes_auto")) or 0
        scored.append(values)

    if not scored:
        print(f"Nothing scored: every row has blanks ({len(rows)} rows).")
        return 1

    present = sum(r["roofs_present"] for r in scored)
    correct = sum(r["roofs_detected_correctly"] for r in scored)
    detected = sum(r["planes_detected_auto"] for r in scored)
    missed = sum(r["missed_lines"] for r in scored)
    false = sum(r["false_lines"] for r in scored)
    spans = sum(r["spans_auto"] for r in scored)
    side_err = sum(r["side_label_errors"] for r in scored)
    edge_err = sum(r["edge_label_errors"] for r in scored)
    front_ok = sum(1 for r in scored if r["front_eave_fully_covered"])
    occl_ok = sum(1 for r in scored if r["occlusion_ok"])
    polygon_planes = sum(r["polygon_path_planes_auto"] for r in scored)

    print("=" * 96)
    print(f"ROOFLINE REVIEW SUMMARY   ({len(scored)} photos scored, {len(skipped)} skipped)")
    print("=" * 96)
    print(f"{'metric':44s} {'value':>14s}   basis")
    print("-" * 96)
    rate = lambda a, b: f"{100.0 * a / b:.1f}%" if b else "n/a"  # noqa: E731
    print(f"{'roof detection recall':44s} {rate(correct, present):>14s}   {correct}/{present} real planes found")
    print(f"{'roof detection precision':44s} {rate(correct, detected):>14s}   {correct}/{detected} detected planes real")
    print(f"{'front eave fully covered':44s} {rate(front_ok, len(scored)):>14s}   {front_ok}/{len(scored)} photos")
    print(f"{'occlusion handled sanely':44s} {rate(occl_ok, len(scored)):>14s}   {occl_ok}/{len(scored)} photos")
    print(f"{'missed lines (total)':44s} {missed:>14d}   {missed / len(scored):.2f} per photo")
    print(f"{'false lines (total)':44s} {false:>14d}   {false / len(scored):.2f} per photo")
    print(f"{'span precision (not false)':44s} {rate(spans - false, spans):>14s}   {spans - false}/{spans} drawn spans")
    print(f"{'span recall (drawn vs drawn+missed)':44s} {rate(spans - false, spans - false + missed):>14s}   "
          f"{spans - false} real / {spans - false + missed} should exist")
    print(f"{'side label accuracy':44s} {rate(spans - side_err, spans):>14s}   {side_err} wrong of {spans}")
    print(f"{'edge label accuracy':44s} {rate(spans - edge_err, spans):>14s}   {edge_err} wrong of {spans}")
    print("-" * 96)
    if polygon_planes:
        print(f"CAVEAT: {polygon_planes} plane(s) took the polygon fallback, which hardcodes")
        print("        side='front' edge='eave' (roofline.py:134-138). Label accuracy above is")
        print("        not meaningful for spans from those planes.")

    print("\nPER-IMAGE")
    print("-" * 96)
    print(f"{'photo':40s} {'real':>5s} {'got':>4s} {'miss':>5s} {'false':>6s} {'side':>5s} {'edge':>5s} "
          f"{'front':>6s} {'occl':>5s}")
    for r in sorted(scored, key=lambda x: (-(x["missed_lines"] + x["false_lines"]), x["photo"])):
        print(f"{r['photo'][:40]:40s} {r['roofs_present']:>5d} {r['roofs_detected_correctly']:>4d} "
              f"{r['missed_lines']:>5d} {r['false_lines']:>6d} {r['side_label_errors']:>5d} "
              f"{r['edge_label_errors']:>5d} {'y' if r['front_eave_fully_covered'] else 'n':>6s} "
              f"{'y' if r['occlusion_ok'] else 'n':>5s}")

    if skipped:
        print(f"\nskipped (blank cells): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
