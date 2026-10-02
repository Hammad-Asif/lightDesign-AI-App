"""M3 acceptance: eave lengths against the tape, on Lukas's measured houses.

Lukas's criterion: the eave length the tool gives is within 10% of the tape on 8 of 10 houses.

Each house is a photo already uploaded and detected in the app. Nothing here detects or calls a
model: it reads the stored photo, its roof lines and its camera, and measures with the same code
the editor's /geometry uses. Free to re-run.

The truth file (JSON), one entry per house, pixels in the stored photo:

    {"houses": [
      {"name": "12 Palm Way",
       "photo_id": "<id from the app's URL>",
       "reference": {"points": [[x, y], [x, y]], "length_in": 192, "facade": null},
       "eaves": [{"points": [[x, y], [x, y]], "tape_in": 480, "facade": null}]}
    ]}

"reference" is the one known length the salesperson would enter (a garage door, a window), on a
different feature from the eaves being judged; "facade" may be left null (read from the vanishing
points). Each eave is marked by its two ends and its taped length. An eave's "facade" is the side
the salesperson would pick in the editor for a line no vanishing point gave a side to (the
template fills in the side detection found); null leaves it to the vanishing points.

Per eave, two numbers:
  geometry  the marked eave through the camera and scale: the perspective and scale math alone.
  app       what the editor shows: the detected eave line that matches the marked one; where
            detection missed it, the marked line, as if the salesperson had drawn it.
A house passes when every eave's app length is within 10% of its tape.

    python tests/measure_eval.py --template <photo_id> [...]   # a skeleton, with each photo's detected eaves
    python tests/measure_eval.py truth.json [--overlays DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))

from PIL import Image, ImageDraw  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import engine  # noqa: E402
from app.models import Photo, Region  # noqa: E402
from app.services import camera, fusion, scale  # noqa: E402

TOLERANCE = 0.10          # Lukas: within 10% of the tape
HOUSES_TO_PASS = 0.8      # ... on 8 of 10 houses
MATCH_REACH = 0.03        # of the long edge: a detected eave's ends this close to the marked one's
MATCH_DEG = 5.0


def load(session: Session, photo_id: str):
    photo = session.get(Photo, photo_id)
    if photo is None:
        raise SystemExit(f"No photo {photo_id} in {get_settings().database_url}")
    regions = list(session.scalars(select(Region).where(Region.photo_id == photo.id)))
    stored = (photo.interpreted or {}).get("camera")
    if stored:
        cam = camera.Camera.from_dict(stored)
    else:
        image = Image.open(get_settings().storage_dir / "photos" / str(photo.id) / "original.png").convert("RGB")
        cam = camera.build(fusion.Context(image, get_settings(), [], None, None).scene, image.size, photo.focal_35mm)
    return photo, regions, cam


def roof_lines(regions) -> list[scale.Line]:
    return [scale.Line(str(r.id), [list(map(float, p)) for p in r.polyline], r.edge, r.side)
            for r in regions if r.geometry == "line" and r.kind == "roof" and r.polyline and len(r.polyline) >= 2]


def _matches(line: scale.Line, marked, reach: float) -> bool:
    a, b = np.asarray(marked[0], float), np.asarray(marked[1], float)
    c, d = np.asarray(line.points[0], float), np.asarray(line.points[-1], float)
    along, other = (b - a) / np.linalg.norm(b - a), (d - c) / max(np.linalg.norm(d - c), 1e-9)
    if np.degrees(np.arccos(min(1.0, abs(float(along @ other))))) > MATCH_DEG:
        return False
    ends = min(np.linalg.norm(a - c) + np.linalg.norm(b - d), np.linalg.norm(a - d) + np.linalg.norm(b - c))
    return ends <= 2 * reach


def evaluate(house: dict, session: Session, overlays: Path | None) -> dict:
    photo, regions, cam = load(session, house["photo_id"])
    reach = MATCH_REACH * max(photo.width, photo.height)
    reference = dict(house["reference"])
    for eave in house["eaves"]:
        if _matches(scale.Line("r", reference["points"], "eave", None), eave["points"], reach):
            raise SystemExit(f"{house['name']}: the reference is the eave being judged; measure another feature")
    lines = roof_lines(regions)
    if not reference.get("facade"):
        reference["facade"] = cam.facade_of(*reference["points"], get_settings().roofline_vp_deg)
    report = {"name": house["name"], "photo_id": house["photo_id"], "focal_source": cam.focal_source,
              "focal_35mm": round(cam.focal_35mm, 1), "eaves": []}
    for index, eave in enumerate(house["eaves"]):
        marked = scale.Line(f"marked{index}", eave["points"], "eave", eave.get("facade"))
        detected = next((line for line in lines if line.edge == "eave" and _matches(line, eave["points"], reach)), None)
        if detected is not None and eave.get("facade"):
            # The side the salesperson gives the line in the editor.
            lines = [scale.Line(l.id, l.points, l.edge, eave["facade"]) if l is detected else l for l in lines]
            detected = next(l for l in lines if l.id == detected.id)
        alone = scale.solve(cam, lines + [marked], reference, photo.interpreted, get_settings().roofline_vp_deg).lines
        geometry = alone[marked.id].length_in
        app_length = alone[detected.id].length_in if detected and alone[detected.id].length_in else geometry
        tape = float(eave["tape_in"])
        error = (app_length / tape - 1) if app_length else None
        report["eaves"].append({
            "tape_in": tape,
            "geometry_in": round(geometry, 1) if geometry else None,
            "app_in": round(app_length, 1) if app_length else None,
            "from": "detected line" if detected and alone[detected.id].length_in else "marked line",
            "error": round(error, 4) if error is not None else None,
            "why_not": alone[marked.id].needs,
            "pass": error is not None and abs(error) <= TOLERANCE,
        })
    report["pass"] = all(e["pass"] for e in report["eaves"])
    if overlays is not None:
        overlay(photo, house, report, overlays)
    return report


def overlay(photo, house: dict, report: dict, directory: Path) -> None:
    image = Image.open(get_settings().storage_dir / "photos" / str(photo.id) / "original.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(3, round(max(photo.width, photo.height) / 400))
    draw.line([tuple(p) for p in house["reference"]["points"]], fill=(245, 158, 11), width=width)
    for eave, result in zip(house["eaves"], report["eaves"]):
        colour = (57, 255, 20) if result["pass"] else (255, 59, 48)
        draw.line([tuple(p) for p in eave["points"]], fill=colour, width=width)
        a, b = eave["points"]
        label = f"tape {result['tape_in'] / 12:.1f} ft, app {result['app_in'] / 12 if result['app_in'] else 0:.1f} ft"
        draw.text(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 4 * width), label, fill=colour)
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / f"{house['name'].replace('/', '_')}.png")


def template(session: Session, photo_ids: list[str]) -> dict:
    houses = []
    for photo_id in photo_ids:
        photo, regions, cam = load(session, photo_id)
        eaves = [line for line in roof_lines(regions) if line.edge == "eave"]
        houses.append({
            "name": "", "photo_id": photo_id,
            "size": [photo.width, photo.height], "lens": cam.focal_source,
            "reference": {"points": [[0, 0], [0, 0]], "length_in": 0, "facade": None},
            "eaves": [{"points": [[round(x), round(y)] for x, y in (line.points[0], line.points[-1])], "tape_in": 0,
                       "facade": line.side}
                      for line in eaves],
        })
    return {"houses": houses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("truth", nargs="?")
    parser.add_argument("--template", nargs="+", metavar="PHOTO_ID")
    parser.add_argument("--overlays", type=Path)
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args()
    with Session(engine()) as session:
        if args.template:
            print(json.dumps(template(session, args.template), indent=2))
            return
        if not args.truth:
            parser.error("give a truth file, or --template with photo ids")
        houses = json.loads(Path(args.truth).read_text())["houses"]
        reports = [evaluate(house, session, args.overlays) for house in houses]
    print(f"{'house':28} {'lens':9} {'tape ft':>8} {'geometry':>9} {'app':>8} {'error':>7}  from")
    for report in reports:
        for eave in report["eaves"]:
            error = f"{100 * eave['error']:+.1f}%" if eave["error"] is not None else eave["why_not"] or "none"
            print(f"{report['name'][:28]:28} {report['focal_source']:9} {eave['tape_in'] / 12:8.1f} "
                  f"{(eave['geometry_in'] or 0) / 12:9.1f} {(eave['app_in'] or 0) / 12:8.1f} {error:>7}  "
                  f"{eave['from']}{'' if eave['pass'] else '  FAIL'}")
    passed = sum(r["pass"] for r in reports)
    needed = int(np.ceil(HOUSES_TO_PASS * len(reports)))
    print(f"\n{passed} of {len(reports)} houses within {TOLERANCE:.0%} of the tape; acceptance needs {needed}: "
          + ("PASS" if passed >= needed else "NOT YET"))
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
