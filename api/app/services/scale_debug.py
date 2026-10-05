"""SCALE_DEBUG: one solve written out step by step, for diagnosing a measurement.

Every /geometry call returns the record (the editor's Scale debug panel) and writes it, with one
overlay per step, into the photo's debug/ folder beside the roofline overlays. Overwritten each
time, so the folder always holds the photo's last solve:

  scale_debug.json         every step's record (scale.Trace) and each line's lifecycle, lines
                           numbered L1, L2 ... as in the overlays
  scale_1_camera.png       vanishing-point guides per facade, the vertical, the focal length and
                           any vanishing-point pair rejected as ill-conditioned
  scale_2_sides.png        each level line's facade and how it was decided (direction and its
                           margin, structure, or ambiguous)
  scale_3_measurement.png  the reference, its wall, and the distance it set
  scale_4_walls.png        each line by the wall it is on (set-back walls too), the corners and
                           roof faces that carried distance between walls
  scale_5_pitch.png        the rakes that measured the pitch, and why others did not
  scale_6_anchors.png      the points ridges hang from, and how sure each is
  scale_7_choice.png       the wall choices tried for undecided lines, and their scores
  scale_8_checks.png       the lines the independent checks flagged
  scale_9_lengths.png      each line's real length, method and confidence

Development only: it never changes a result.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.camera import Camera
from app.services.scale import Line, Solution, Trace

FACADE_COLOURS = {"front": (0, 200, 255), "left": (255, 80, 200), "right": (120, 230, 60)}
EDGE_COLOURS = {"eave": (0, 200, 255), "rake": (255, 159, 10), "ridge": (255, 60, 60), "hip": (190, 90, 255),
                "valley": (60, 120, 255), "junction": (120, 230, 60)}
CONFIDENCE_COLOURS = {"exact": (40, 210, 90), "stated": (255, 190, 0), "assumed": (255, 90, 40)}
GREY = (170, 170, 170)
RED = (255, 50, 50)


def _clean(value):
    """JSON-safe: numpy to Python, infinities and NaN to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _feet(inches) -> str:
    if inches is None:
        return "?"
    total = round(abs(float(inches)))
    return f"{total // 12}' {total % 12}\""


def _wall_colour(key: str | None) -> tuple:
    if not key:
        return GREY
    facade, _, setback = key.partition("+")
    facade = facade.partition("#")[0].partition(".")[0].partition("@")[0]
    colour = FACADE_COLOURS.get(facade, GREY)
    if setback:
        # A set-back wall: its facade's colour, lighter.
        return tuple(int(c + (255 - c) * 0.55) for c in colour)
    return colour


def record(camera: Camera, lines: list[Line], scale: dict | None, trace: Trace, photo_id: str,
           labels: dict | None = None) -> dict:
    """The JSON the editor shows and scale_debug.json holds."""
    number = {line.id: f"L{i}" for i, line in enumerate(lines, 1)}
    return _clean({
        "photo_id": photo_id, "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scale": scale, "steps": trace.steps, "numbers": number,
        "lines": [{"n": number.get(entry["id"]), "label": (labels or {}).get(entry["id"]), **entry}
                  for entry in trace.lines.values()],
    })


class _Canvas:
    def __init__(self, image: Image.Image, dim: float = 0.0):
        self.image = image.copy().convert("RGB")
        if dim:
            self.image = Image.blend(self.image, Image.new("RGB", self.image.size, (0, 0, 0)), dim)
        self.draw = ImageDraw.Draw(self.image)
        long_edge = max(self.image.size)
        self.font = ImageFont.load_default(size=max(14, long_edge // 70))
        self.width = max(2, long_edge // 400)

    def line(self, points, colour, width=None):
        self.draw.line([tuple(map(float, p)) for p in points], fill=colour, width=width or self.width)

    def dot(self, point, colour, radius=None):
        r = radius or self.width * 3
        x, y = map(float, point)
        self.draw.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=self.width)

    def label(self, point, text, colour):
        x, y = map(float, point)
        box = self.draw.textbbox((x, y), text, font=self.font)
        self.draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0))
        self.draw.text((x, y), text, fill=colour, font=self.font)

    def panel(self, rows: list[str]):
        y = 8
        for row in rows:
            self.label((8, y), row, (255, 255, 255))
            y += self.font.size + 8

    def save(self, path: Path):
        self.image.save(path)


def _middle(points):
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    return (a + b) / 2


def _vp_guides(canvas: _Canvas, vp, colour, size):
    """Rays from the photo's edges toward a vanishing point (or along its direction at infinity)."""
    width, height = size
    starts = [(0, y) for y in np.linspace(0, height, 7)] + [(width, y) for y in np.linspace(0, height, 7)]
    if vp[2] == 0.0:
        direction = np.asarray(vp[:2], float)
        for s in starts:
            s = np.asarray(s, float)
            canvas.line([s - direction * 4 * width, s + direction * 4 * width], colour, 1)
        return
    target = np.asarray(vp[:2], float) / vp[2]
    for s in starts:
        canvas.line([s, target], colour, 1)


def write(image: Image.Image, camera: Camera, lines: list[Line], solution: Solution, trace: Trace,
          record_: dict, directory: Path) -> list[str]:
    """Writes the record and the overlays; returns the file names written."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scale_debug.json").write_text(json.dumps(record_, indent=2))
    written = ["scale_debug.json"]
    number = record_["numbers"]
    size = image.size
    placed = solution.lines
    steps = trace.steps
    info = trace.lines

    def save(canvas: _Canvas, name: str):
        canvas.save(directory / name)
        written.append(name)

    # 1 camera
    canvas = _Canvas(image)
    for facade, vp in zip(camera.facades, camera.horizontal):
        _vp_guides(canvas, vp, FACADE_COLOURS.get(facade, GREY), size)
    if camera.vertical is not None:
        _vp_guides(canvas, camera.vertical, (255, 255, 255), size)
    cam = steps.get("1_camera", {})
    rows = [f"focal {camera.focal:.0f}px = {camera.focal_35mm:.1f}mm ({camera.focal_source})",
            "facades " + ", ".join(camera.facades) + "   (white = vertical)",
            f"used for walls: {', '.join(cam.get('facades_used') or [])}"
            + (f"   dropped: {cam['facade_dropped']} (no house line runs toward it)" if cam.get("facade_dropped") else "")]
    for rejected in camera.focal_checks.get("vp_rejected", []):
        rows.append(f"rejected pair: {rejected.get('focal_35mm')}mm, spread {rejected.get('spread')}")
    rows += [f"{k}: {v}" for k, v in camera.focal_checks.items() if k not in ("vp_rejected",)]
    canvas.panel(rows)
    save(canvas, "scale_1_camera.png")

    # 2 sides
    canvas = _Canvas(image)
    sides = steps.get("2_sides", {}).get("lines", {})
    for line in lines:
        found = sides.get(line.id)
        if found is None:
            canvas.line(line.points, GREY, 1)
            continue
        colour = FACADE_COLOURS.get(found.get("facade"), GREY) if found.get("how") != "ambiguous" else (255, 255, 255)
        chosen = placed.get(line.id).side if placed.get(line.id) else None
        canvas.line(line.points, colour)
        margin = found.get("margin_deg")
        text = (f"{number[line.id]} {found.get('facade') or chosen or '?'} by {found.get('how')}"
                + (f" ({margin:.1f} deg)" if margin is not None else "")
                + (f" -> chose {chosen} of {'/'.join(found.get('candidates') or [])}" if found.get("how") == "ambiguous" else ""))
        canvas.label(_middle(line.points), text, colour)
    canvas.panel([f"side of each level line: direction needs a {steps.get('2_sides', {}).get('margin_min_deg')} deg margin "
                  "over the next vanishing point; white = ambiguous (tried both)"])
    save(canvas, "scale_2_sides.png")

    # 3 measurement
    canvas = _Canvas(image)
    for line in lines:
        canvas.line(line.points, GREY, 1)
    step = steps.get("3_measurement", {})
    pixels = step.get("pixels")
    if pixels:
        canvas.line(pixels, (255, 0, 255), canvas.width * 2)
        for p in (pixels[0], pixels[-1]):
            canvas.dot(p, (255, 0, 255))
        canvas.label(_middle(pixels), f"{_feet(step.get('length_in'))} on {step.get('wall')}", (255, 0, 255))
    found = step.get("side_assignment") or {}
    rows = [f"reference by {step.get('by')}; side {step.get('facade')} by {step.get('how')}"
            + (f" (via {number.get(found.get('via'), found.get('via'))})" if found.get("via") else "")
            + f"; wall {step.get('wall')}; on eave plane: {step.get('on_eave')}"]
    if "wall_distance_in" in step:
        rows.append(f"measured plane {_feet(step['measured_plane_in'])} from camera + overhang "
                    f"{step['overhang_added_in']:g}in = wall {_feet(step['wall_distance_in'])}")
    if step.get("needs") or solution.needs:
        rows.append(f"needs: {step.get('needs') or solution.needs}")
    canvas.panel(rows)
    save(canvas, "scale_3_measurement.png")

    # 4 walls
    canvas = _Canvas(image)
    for line in lines:
        key = info.get(line.id, {}).get("wall") or (placed.get(line.id).wall if placed.get(line.id) else None)
        colour = _wall_colour(key)
        canvas.line(line.points, colour)
        canvas.label(_middle(line.points), f"{number[line.id]} {key or 'no wall'}", colour)
    walls = steps.get("4_walls", {})
    for corner in walls.get("corners", []):
        canvas.dot(corner["corner_px"], _wall_colour(corner["to_wall"]))
    rows = ["walls (distance from camera; lighter = set back on a roof)"]
    rows += [f"{k}: {_feet(w['distance_in'])} ({w['source']}{', via pitch' if w.get('via_pitch') else ''})"
             for k, w in solution.walls.items()]
    rows += [f"set-back {s['key']}: foot {number.get(s['junction'])}, from eave {number.get(s['eave'])}, "
             f"members {', '.join(number.get(m, '?') for m in s['members'])}" for s in walls.get("setbacks", [])]
    rows += [f"roof-face link {link['from']} -> {link['to']}: {_feet(link['distance_in'])}" for link in walls.get("links", [])]
    rows += [f"rejected {r['wall']}: {r['why']}" for r in walls.get("rejected", [])]
    if walls.get("unreached"):
        rows.append("unreached: " + ", ".join(walls["unreached"]))
    canvas.panel(rows + ["circles: corners that passed a distance on"])
    save(canvas, "scale_4_walls.png")

    # 5 pitch
    canvas = _Canvas(image)
    pitch = steps.get("5_pitch", {})
    by_id = {line.id: line for line in lines}
    for line in lines:
        canvas.line(line.points, GREY, 1)
    for rake in pitch.get("rakes", []):
        line = by_id.get(rake["id"])
        if line is None:
            continue
        colour = (40, 210, 90) if rake.get("used") else RED
        canvas.line(line.points, colour)
        text = f"{number[line.id]} " + (f"{rake['rise']}/12" + (" peak" if rake.get("at_peak") else " lone")
                                        if rake.get("used") else f"not used: {rake.get('why')}")
        canvas.label(_middle(line.points), text, colour)
    used = pitch.get("used") or {}
    canvas.panel([f"pitch {used.get('rise')}/12 ({used.get('source')}); photo's own estimate {pitch.get('image_estimate')}",
                  "green: rakes that measured it, red: rakes that could not"])
    save(canvas, "scale_5_pitch.png")

    # 6 anchors
    canvas = _Canvas(image)
    for line in lines:
        done = placed.get(line.id)
        colour = EDGE_COLOURS.get(line.edge, GREY)
        canvas.line(line.points, colour, 1)
        anchors = info.get(line.id, {}).get("ridge_anchors")
        if anchors is not None:
            canvas.label(_middle(line.points), f"{number[line.id]} ends on {anchors[0] or '-'} / {anchors[1] or '-'}"
                         + (f" ({done.method})" if done else ""), colour)
    for anchor in steps.get("6_anchors", {}).get("anchors", []):
        colour = CONFIDENCE_COLOURS.get(anchor.get("sure"), GREY)
        canvas.dot(anchor["px"], colour, canvas.width * 5)
        canvas.label((anchor["px"][0] + 8, anchor["px"][1] - 24), f"{anchor['what']} ({anchor.get('sure')})", colour)
    canvas.panel(["points ridges hang from, by confidence"])
    save(canvas, "scale_6_anchors.png")

    # 7 choice
    choice = steps.get("7_choice", {})
    if choice.get("items"):
        canvas = _Canvas(image, dim=0.55)
        rows = ["undecided: " + "; ".join(f"{number.get(item['id'], 'reference')} {'/'.join(item['candidates'])}"
                                          for item in choice["items"])]
        for tried in sorted(choice.get("tried", []), key=lambda t: t["score"])[:12]:
            picks = ", ".join(f"{number.get(k, 'reference')}={v}" for k, v in tried["choice"].items())
            mark = "  <- kept" if tried["choice"] == choice.get("chosen") else ""
            rows.append(f"score {tried['score']:.2f} ({tried['checks']} checks): {picks}{mark}")
        canvas.panel(rows)
        save(canvas, "scale_7_choice.png")

    # 8 checks
    canvas = _Canvas(image)
    flagged = {line_id for check in solution.checks for line_id in check.get("lines", [])}
    for line in lines:
        canvas.line(line.points, RED if line.id in flagged else GREY, canvas.width * (2 if line.id in flagged else 1))
        if line.id in flagged:
            canvas.label(_middle(line.points), number[line.id], RED)
    feature_rows = [f"{f.get('label') or f.get('kind')}: wall {f.get('wall')}, "
                    + (f"{_feet(f['measured']['height_in'])} x {_feet(f['measured']['width_in'])}" if f.get("measured") else "not measured")
                    for f in steps.get("8_checks", {}).get("features", [])]
    canvas.panel([f"checks (penalty {steps.get('8_checks', {}).get('penalty')}):"]
                 + [f"[{c['kind']}] {c['message'][:150]}" for c in solution.checks] + feature_rows
                 + ([] if solution.checks else ["all passed"]))
    save(canvas, "scale_8_checks.png")

    # 9 lengths
    canvas = _Canvas(image)
    for line in lines:
        done = placed.get(line.id)
        colour = CONFIDENCE_COLOURS.get(done.confidence, GREY) if done else GREY
        canvas.line(line.points, colour)
        text = (f"{number[line.id]} {line.edge} {_feet(done.length_in)} {done.method}" if done and done.length_in
                else f"{number[line.id]} needs {done.needs if done else '?'}")
        canvas.label(_middle(line.points), text, colour)
    pitch = solution.pitch or {}
    canvas.panel([f"pitch {pitch.get('rise')}/12 ({pitch.get('source')})",
                  "green exact, amber stated, orange assumed, grey not placed"]
                 + ([f"warning: {solution.warning[:150]}"] if solution.warning else []))
    save(canvas, "scale_9_lengths.png")
    for stale in directory.glob("scale_*.png"):
        if stale.name not in written:
            stale.unlink()
    return written
