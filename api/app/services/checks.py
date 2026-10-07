"""Checks of a scale solution against what the solve did not produce (M3: scale).

A flag that compares a solve with itself catches nothing: a wrong wall moves the value and the
suggestion together. These compare against outside knowledge instead:

  - size: detected doors and garage doors come out a standard size on their wall;
  - meets: a line ending at a wall line ends on that wall, not feet in front of or behind it
    (an eave's overhang allowed);
  - pitch: hips and rakes imply a pitch a house can have;
  - eave_height: eaves sit at a height above a standing camera that a house can have;
  - plausibility: a one-storey eave run is not longer than a house, and no wall is many times
    the measured one; nor is the whole house smaller than any house (a reference typed too
    small shrinks every wall and gable alike, so it is said, never scored).

Each failed check has a penalty; scale.solve keeps the wall choice with the least. The checks of
the kept choice are shown: a flag means no choice satisfied them.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.services.camera import Camera

STANDARD = {
    "garage_door": {"name": "a garage door", "height": (78.0, 108.0), "width": (84.0, 228.0),
                    "typical": "7 to 8 ft tall, 8 to 9 ft wide (single) or 16 ft (double)"},
    # Width up to a 16 ft sliding patio door: "door" covers those too, and only the height is standard.
    "door": {"name": "a door", "height": (74.0, 102.0), "width": (26.0, 200.0),
             "typical": "about 6 ft 8 in tall"},
}
# Heights the doors are built to (in): entry and patio doors 80 in (some 84 or 96), garage doors 7 or
# 8 ft. Heights, not widths: a garage is 8 or 16 ft wide, a patio door 5 to 16, but nearly every
# door is about as tall as every other.
DOOR_HEIGHT = {"door": (78.0, 96.0), "garage_door": (84.0, 96.0)}
# The doors' scale this far from the typed length's (either way) is said: past the spread of door
# heights, the typed length or what it was tapped on is the likelier mistake.
SCALE_CONFLICT = 0.20
PLAUSIBLE_PITCH = (2.0, 18.0)          # rise per 12
ONE_STOREY_EAVE_FT = 80.0              # eaves this low above the camera are one storey
ONE_STOREY_HEIGHT_FT = 8.0
ANY_EAVE_FT = 100.0
WALL_RATIO = 3.0
EAVE_HEIGHT_FT = (-4.0, 35.0)          # above a standing camera
OFF_WALL_IN = 24.0                     # a line meeting a wall may end this far off it, beyond overhangs
SIZE_SLACK = 0.10                      # sizes this far past their range are flagged, not only penalised
# Smaller than a house: no wall wider than this, and no gable rake longer than SMALL_RAKE_FT.
# Measured: a 12 in reference on 824fd558 made a two-storey front 18.6 ft and every rake 5 ft
# 6 in or under; the other measured photos each had a rake of 6 ft or more. Without a gable the
# width alone says nothing: a photo's only "eave" can be a scrap a foot long (38623ee6, df1a331b).
SMALL_HOUSE_FT = 20.0
SMALL_RAKE_FT = 6.0


@dataclass
class Context:
    camera: Camera
    lines: list
    placed: dict                       # line id -> scale.Placed
    walls: dict                        # wall key -> {"distance_in", "facade", ...}
    wall_of: dict                      # line id -> wall key
    normals: dict                      # wall key -> unit normal
    up: np.ndarray
    measured: str | None               # the measured wall's key
    reach: float
    overhang: Callable                 # line -> inches its plane sits in front of its wall
    features: list = field(default_factory=list)   # {"id", "kind", "label", "wall", "top", "bottom", "left", "right"}
    corner_rejections: int = 0


def _feet(inches: float) -> str:
    inches = abs(float(inches))
    if inches < 24:
        return f"{inches:.0f} in"
    feet, rest = divmod(round(inches), 12)
    return f"{feet} ft {rest} in" if rest else f"{feet} ft"


def _on_wall(ctx: Context, pixel, key: str) -> np.ndarray | None:
    normal, distance = ctx.normals[key], ctx.walls[key]["distance_in"]
    point = ctx.camera.on_wall(pixel, normal, distance)
    return point if point is not None and float(point[2]) > 0 else None


def _outside(value: float, low: float, high: float) -> float:
    """How far past [low, high] a value is, as a log ratio; 0 inside."""
    if value < low:
        return float(np.log(low / max(value, 1e-6)))
    if value > high:
        return float(np.log(value / high))
    return 0.0


def evaluate(ctx: Context) -> tuple[float, list[dict]]:
    penalty = 0.5 * ctx.corner_rejections
    found: list[dict] = []
    by_id = {line.id: line for line in ctx.lines}

    # size: standard objects on their wall.
    for feature in ctx.features:
        rule = STANDARD.get(feature["kind"])
        key = feature.get("wall")
        if rule is None or key not in ctx.walls:
            continue
        top, bottom = _on_wall(ctx, feature["top"], key), _on_wall(ctx, feature["bottom"], key)
        left, right = _on_wall(ctx, feature["left"], key), _on_wall(ctx, feature["right"], key)
        if any(p is None for p in (top, bottom, left, right)):
            continue
        height, width = float(np.linalg.norm(top - bottom)), float(np.linalg.norm(right - left))
        feature["measured"] = {"height_in": round(height, 1), "width_in": round(width, 1)}
        off = _outside(height, *rule["height"]) + 0.5 * _outside(width, *rule["width"])
        penalty += 4.0 * off
        if off > np.log(1 + SIZE_SLACK):
            found.append({"kind": "size", "lines": [feature["id"]],
                          "message": f"{feature.get('label') or rule['name'].capitalize()} measures {_feet(height)} tall "
                                     f"and {_feet(width)} wide here; {rule['name']} is {rule['typical']}. "
                                     "Check the measurement, or the side of the house it is on."})

    # meets: a line ending at a wall line's end ends on that wall.
    wall_ends = []
    for line in ctx.lines:
        key = ctx.wall_of.get(line.id)
        done = ctx.placed.get(line.id)
        if key in ctx.walls and done is not None and done.points3d and line.edge in ("eave", "junction"):
            wall_ends += [(line, key, line.points[0]), (line, key, line.points[-1])]
    for line in ctx.lines:
        done = ctx.placed.get(line.id)
        if done is None or not done.points3d or done.method == "override":
            continue
        for index, end in ((0, line.points[0]), (-1, line.points[-1])):
            for other, key, tip in wall_ends:
                if other is line or ctx.wall_of.get(line.id) == key:
                    continue
                if float(np.hypot(end[0] - tip[0], end[1] - tip[1])) > ctx.reach:
                    continue
                distance = ctx.walls[key]["distance_in"]
                along = float(np.sign(distance)) * float(ctx.normals[key] @ done.points3d[index]) - abs(distance)
                allowed = OFF_WALL_IN + ctx.overhang(line) + ctx.overhang(other)
                if abs(along) > allowed:
                    penalty += float(np.log1p((abs(along) - allowed) / OFF_WALL_IN))
                    where = "in front of" if along < 0 else "behind"
                    found.append({"kind": "meets", "lines": [line.id, other.id],
                                  "message": f"A {line.edge} ends {_feet(abs(along))} {where} the "
                                             f"{ctx.walls[key].get('facade', key)} wall it meets."})
                break

    # pitch: hips and gable rakes.
    for line in ctx.lines:
        done = ctx.placed.get(line.id)
        if done is None or not done.points3d or done.method not in ("hip", "gable"):
            continue
        v = done.points3d[-1] - done.points3d[0]
        rise = abs(float(v @ ctx.up))
        run = float(np.linalg.norm(v - (v @ ctx.up) * ctx.up))
        if done.method == "hip":
            run /= np.sqrt(2)
        pitch = 12.0 * rise / run if run > 1e-9 else float("inf")
        if not PLAUSIBLE_PITCH[0] <= pitch <= PLAUSIBLE_PITCH[1]:
            penalty += 1.0
            found.append({"kind": "pitch", "lines": [line.id],
                          "message": f"A {line.edge} placed here implies a {pitch:.0f}/12 roof pitch, "
                                     "outside a house's usual range." if np.isfinite(pitch)
                                     else f"A {line.edge} placed here stands vertical."})

    # eave height above the camera.
    for line in ctx.lines:
        done = ctx.placed.get(line.id)
        if line.edge != "eave" or done is None or not done.points3d:
            continue
        height = float(np.mean([p @ ctx.up for p in done.points3d])) / 12.0
        if height < EAVE_HEIGHT_FT[0]:
            off = (EAVE_HEIGHT_FT[0] - height) / 10.0
        elif height > EAVE_HEIGHT_FT[1]:
            off = float(np.log(height / EAVE_HEIGHT_FT[1]))
        else:
            off = 0.0
        if off > 0:
            penalty += 2.0 * off
            found.append({"kind": "eave_height", "lines": [line.id],
                          "message": f"An eave comes out {height:.0f} ft above the camera; "
                                     "a photo taken standing puts eaves about 2 to 25 ft above it."})

    # plausibility: eave runs per wall.
    spans = {}
    for key in ctx.walls:
        runs = [done.points3d for line in ctx.lines if ctx.wall_of.get(line.id) == key and line.edge in ("eave", "junction")
                for done in [ctx.placed.get(line.id)] if done is not None and done.points3d
                and done.method in ("wall", "measured")]
        if not runs:
            continue
        points = [p for run in runs for p in run]
        longest = max(runs, key=lambda run: float(np.linalg.norm(run[-1] - run[0])))
        level = longest[-1] - longest[0]
        level = level - (level @ ctx.up) * ctx.up
        if np.linalg.norm(level) < 1e-9:
            continue
        level = level / np.linalg.norm(level)
        along = [float(p @ level) for p in points]
        height = float(np.mean([p @ ctx.up for p in points])) / 12.0
        spans[key] = ((max(along) - min(along)) / 12.0, height)
    for key, (feet, height) in spans.items():
        limit = ONE_STOREY_EAVE_FT if height < ONE_STOREY_HEIGHT_FT else ANY_EAVE_FT
        facade = ctx.walls[key].get("facade", key)
        if feet > limit:
            penalty += 2.0 * float(np.log(feet / limit))
            found.append({"kind": "plausibility", "lines": [],
                          "message": f"This measurement makes the {facade} of the house {feet:.0f} ft wide, which is "
                                     "more than a house usually is. Check the length you typed, or measure something "
                                     "of a known size instead: a single garage door is about 9 ft, a double 16 ft, "
                                     "an entry door 80 in tall."})
    # plausibility, the other way: a known length typed too small (or tapped on too long a
    # feature) shrinks the whole house. Each side's eave runs taken together (its separate walls,
    # "front" and "front.2", and its set-back walls alike), so a side split into walls is not
    # read as narrow.
    sides: dict[str, list] = {}
    for line in ctx.lines:
        key, done = ctx.wall_of.get(line.id), ctx.placed.get(line.id)
        if (key in ctx.walls and line.edge in ("eave", "junction") and done is not None and done.points3d
                and done.method in ("wall", "measured")):
            sides.setdefault(ctx.walls[key].get("facade", key), []).append(done.points3d)
    widest = max((feet for feet, _ in spans.values()), default=0.0)
    for runs in sides.values():
        longest = max(runs, key=lambda run: float(np.linalg.norm(run[-1] - run[0])))
        level = longest[-1] - longest[0]
        level = level - (level @ ctx.up) * ctx.up
        if np.linalg.norm(level) < 1e-9:
            continue
        level = level / np.linalg.norm(level)
        along = [float(p @ level) for run in runs for p in run]
        widest = max(widest, (max(along) - min(along)) / 12.0)
    # Without a gable the width alone says nothing (SMALL_RAKE_FT): both must be small.
    rakes = [float(np.linalg.norm(done.points3d[-1] - done.points3d[0])) / 12.0
             for line in ctx.lines if line.edge == "rake"
             for done in [ctx.placed.get(line.id)] if done is not None and done.points3d and done.method == "gable"]
    if spans and rakes and widest < SMALL_HOUSE_FT and max(rakes) < SMALL_RAKE_FT:
        # Said, not scored: every wall choice shrinks alike, so it would decide nothing.
        found.append({"kind": "plausibility", "lines": [],
                      "message": f"This measurement makes the house at most {widest:.0f} ft wide"
                                 + f" and its longest gable rake {_feet(12.0 * max(rakes))}"
                                 + ", which is smaller than a house usually is. Check the length you typed, or measure "
                                   "something of a known size instead: a single garage door is about 9 ft, a double "
                                   "16 ft, an entry door 80 in tall."})
    measured = spans.get(ctx.measured)
    if measured and measured[0] >= 8.0:
        for key, (feet, _) in spans.items():
            if key != ctx.measured and feet > WALL_RATIO * measured[0]:
                penalty += float(np.log(feet / (WALL_RATIO * measured[0])))
                found.append({"kind": "plausibility", "lines": [],
                              "message": f"The {ctx.walls[key].get('facade', key)} wall comes out "
                                         f"{feet / measured[0]:.1f} times as wide as the measured one. "
                                         "Check the measurement and the side it is on."})
    return penalty, found


def scale_evidence(features: list, typed_in: float) -> dict | None:
    """The doors' heights as a second opinion on the scale. Each door measured on its wall says how
    far the scale would have to move for it to stand a standard height (1 when it already does);
    their median against the typed length. A conflict is a warning only: the typed length stands.
    A door whose bottom is hidden (a car, a shrub) reads short, so a lone short door is said
    gently. None when no door was measured."""
    sources = []
    for feature in features:
        heights = DOOR_HEIGHT.get(feature.get("kind"))
        measured = (feature.get("measured") or {}).get("height_in")
        if heights is None or not measured or measured <= 0:
            continue
        low, high = heights
        factor = 1.0 if low <= measured <= high else (low / measured if measured < low else high / measured)
        sources.append({"id": feature["id"], "kind": feature["kind"], "label": feature.get("label"),
                        "height_in": round(float(measured), 1), "factor": round(float(factor), 3)})
    if not sources:
        return None
    factor = float(np.exp(np.median([np.log(s["factor"]) for s in sources])))
    suggested = typed_in * factor
    out = {"typed_in": round(float(typed_in), 1), "factor": round(factor, 3), "suggested_in": round(suggested, 1),
           "sources": sources, "conflict": False, "message": None}
    if abs(np.log(factor)) <= np.log(1.0 + SCALE_CONFLICT):
        return out
    out["conflict"] = True
    shown = ", ".join(f"{s['label'] or STANDARD[s['kind']]['name']} {_feet(s['height_in'])}" for s in sources[:3])
    which = "door" if len(sources) == 1 else "doors"
    out["message"] = (f"Your measurement ({_feet(typed_in)}) makes the {which} {'too tall' if factor < 1 else 'too short'}: "
                      f"{shown}, where doors are about 6 ft 8 in and garage doors 7 to 8 ft. The {which} suggest your "
                      f"length is nearer {_feet(suggested)}. Check the length you typed and what you measured"
                      + (", unless a door's bottom is hidden" if factor > 1 else "") + ".")
    return out
