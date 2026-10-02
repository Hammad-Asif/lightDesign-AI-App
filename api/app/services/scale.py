"""Real scale and 3-D position for every roof line of a photo (M3 Day 2).

Everything here is in the camera's frame (services/camera.py), in inches: the salesperson's one
measurement fixes how far its wall is from the camera, and from there every line is placed in
3-D by a camera ray meeting a plane. What each line needed to assume is recorded with it.

Walls. A facade is a vertical plane n·X = d; camera.py gives n from the vanishing points. The
measured wall's d is the one that makes the measurement come out at its real length. Another
wall inherits d through a corner it shares with a known one: where an eave or rake of each meet,
the point is on both walls, so it is fixed in 3-D by the known wall and the other wall's plane
must pass through it. Walls no corner reaches stay unknown ("needs a measurement").

Lines, by edge:
  - eave, junction: on the wall of their facade. (The eave's overhang puts it a foot or two in
    front of the wall it is measured against; at house distances that is a few percent, and
    lengths measured along the eave itself are unaffected.)
  - rake: on its gable wall, never the wall of the eave it meets. The gable wall is the facade
    whose level direction the line joining the two rake feet runs toward; with one rake in view,
    the facade other than its eave's. A "rake" no second rake meets at a peak, rising from an
    eave's end to a ridge's or hip's end, is a hip roof's corner that detection mislabelled: it
    is placed by rule (b), and says so. Drawing the other rake makes it a gable again.
  - ridge, hip, valley, by the first rule that applies, each stating its assumption:
      (a) a ridge ending at a gable peak (two placed rakes meeting) or at a hip top. Anchored at
          both ends, it runs between the two; at one, level from it toward its facade's vanishing
          point. Exact only from gable peaks; from a hip top, as sure as the hip.
      (b) a hip rising from an eave corner: adjacent faces share one pitch, so the hip lies over
          the corner's bisector in plan; its top is the camera ray meeting that vertical plane.
      (c) otherwise: the roof face rises from the nearest placed eave at the photo's estimated
          standard pitch, or 6/12 without one, and the line's points are rays meeting that face.
  - anything else (an unknown edge, a line with no facade and no anchor): the salesperson's
    override, a midpoint handle and a typed length (override_fraction), or "needs".
"""

from dataclasses import dataclass, field

import numpy as np

from app.services.camera import Camera

CORNER_REACH_FRACTION = 0.02    # of the photo's long edge: line ends this close are one corner
DEFAULT_PITCH_RISE = 6.0        # per 12 of run, when the photo gives no estimate
IMPLAUSIBLE_PITCH = (1.0, 24.0)  # rise per 12: a hip outside this breaks rule (b)'s assumption
LEVEL = {"eave", "junction", "ridge"}
RAKE_PAIR_RATIO = 2.0           # two rakes this unequal in the photo are not one gable's two sides
TYPED_LENGTH_RATIO = 2.0        # a typed length this far off what the photo suggests is questioned
# A house's eaves along the measured wall span at most this, in feet. Wider, and the reference is
# more likely wrong than the house (a window typed as 16 ft made a one-storey front 112 ft). No
# lower bound: the eaves in view may be one short piece, which says nothing about the house.
PLAUSIBLE_WALL_FT = 100.0

EXACT, STATED, ASSUMED = "exact", "stated", "assumed"


@dataclass
class Line:
    id: str
    points: list[list[float]]   # photo pixels
    edge: str | None
    side: str | None


@dataclass
class Placed:
    id: str
    points3d: list[np.ndarray] | None = None
    length_in: float | None = None
    plane: str | None = None        # which surface it was placed on, in words
    method: str = "none"            # wall | gable | ridge | hip | pitch | override | none
    assumption: str | None = None
    confidence: str | None = None   # exact | stated | assumed
    needs: str | None = None        # why it could not be placed
    wall: str | None = None         # the facade whose wall plane holds it (eaves, rakes)
    warning: str | None = None      # a typed length the photo disagrees with

    def to_dict(self) -> dict:
        return {"wall": self.wall, "warning": self.warning, 
            "points3d": [[round(float(v), 2) for v in p] for p in self.points3d] if self.points3d else None,
            "length_in": round(self.length_in, 1) if self.length_in is not None else None,
            "plane": self.plane, "method": self.method, "assumption": self.assumption,
            "confidence": self.confidence, "needs": self.needs,
        }


@dataclass
class Solution:
    walls: dict = field(default_factory=dict)   # facade -> {"distance_in", "source"}
    lines: dict = field(default_factory=dict)   # line id -> Placed
    needs: str | None = None
    warning: str | None = None                  # the measurement makes the house implausible, in words

    def to_dict(self) -> dict:
        return {"walls": self.walls, "lines": {k: v.to_dict() for k, v in self.lines.items()}, "needs": self.needs,
                "warning": self.warning}


# -- small geometry ---------------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else v


def _length(points3d: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points3d[:-1], points3d[1:])))


def _on_plane(camera: Camera, pixel, normal: np.ndarray, offset: float) -> np.ndarray | None:
    """Where the pixel's ray meets the plane, or None when that is behind the camera: a plane the
    ray meets only behind the lens is not what the photo shows at that pixel."""
    point = camera.on_wall(pixel, normal, offset)
    return point if point is not None and float(point[2]) > 0 else None


def _closest_on_line(camera: Camera, pixel, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """The point of the 3-D line origin + s·direction nearest the pixel's camera ray."""
    ray = camera.ray(pixel)
    # Minimise |origin + s·d − t·ray| over s, t.
    a, b, c = float(direction @ direction), float(direction @ ray), float(ray @ ray)
    w = origin
    d, e = float(direction @ w), float(ray @ w)
    denominator = a * c - b * b
    s = (b * e - c * d) / denominator if abs(denominator) > 1e-12 else 0.0
    return origin + s * direction


def world_up(camera: Camera) -> np.ndarray:
    """The vertical, pointing up in the world: its image runs up the photo from the centre."""
    up = camera.up()
    return -up if up[1] > 0 else up


def pitch_hint(interpreted: dict | None) -> tuple[float, str]:
    """(rise per 12, where from): the median standard pitch the photo's planes gave, else 6/12."""
    rises = []
    for entry in ((interpreted or {}).get("pitch") or {}).values():
        standard = (entry or {}).get("nearest_standard") or {}
        if standard.get("rise"):
            rises.append(float(standard["rise"]))
    if rises:
        return float(np.median(rises)), "estimated"
    return DEFAULT_PITCH_RISE, "assumed"


def override_fraction(t: float, u_mid: float) -> float:
    """The client's midpoint mapping: the fraction u along the drawn line of the point a fraction
    t of the way along the real edge, given that the real midpoint was placed at u_mid.
    u = V·t / (t + V − 1), V = u_mid / (2·u_mid − 1). u_mid = 0.5 is even spacing."""
    if abs(u_mid - 0.5) < 1e-6:
        return t
    V = u_mid / (2 * u_mid - 1)
    return V * t / (t + V - 1)


# -- the solve -----------------------------------------------------------------------------------------


def _ends_meet(a: list[float], b: list[float], reach: float) -> bool:
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) <= reach


def _rake_pairs(rake: Line, lines: list[Line], reach: float) -> list[tuple[list[float], list[float]]]:
    """The gable peaks a rake makes with other rakes: (its foot, the other rake's foot) for each."""
    ends = [rake.points[0], rake.points[-1]]
    span = float(np.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]))
    pairs = []
    for other in lines:
        if other is rake or other.edge != "rake":
            continue
        # A gable's two rakes are of a size and meet at the top of both. A broken-off piece of the
        # other slope is neither, and the line between the "feet" would then point anywhere.
        their_span = float(np.hypot(other.points[-1][0] - other.points[0][0], other.points[-1][1] - other.points[0][1]))
        if not span or not their_span or max(span, their_span) / min(span, their_span) > RAKE_PAIR_RATIO:
            continue
        for i, mine in enumerate(ends):
            for theirs, their_other in ((other.points[0], other.points[-1]), (other.points[-1], other.points[0])):
                apex = mine[1] <= ends[1 - i][1] and theirs[1] <= their_other[1]
                if apex and _ends_meet(mine, theirs, reach):
                    pairs.append((ends[1 - i], their_other))
    return pairs


def _reads_as_hip(rake: Line, lines: list[Line], reach: float) -> bool:
    """A rake rising from an eave's end to a ridge's or hip's end, with no other rake ending at its
    top, is a hip roof's corner. A gable's peak always has the other slope coming down from it,
    even when detection broke that slope into pieces of any size; a hip roof's corner never does."""
    ends = [rake.points[0], rake.points[-1]]
    top = 0 if ends[0][1] <= ends[1][1] else 1

    def meets(end, edges) -> bool:
        return any(other is not rake and other.edge in edges and
                   any(_ends_meet(end, tip, reach) for tip in (other.points[0], other.points[-1]))
                   for other in lines)

    return (not meets(ends[top], ("rake",)) and meets(ends[1 - top], ("eave", "junction"))
            and meets(ends[top], ("ridge", "hip")))


def _gable_facade(camera: Camera, rake: Line, lines: list[Line], reach: float, tolerance: float) -> str | None:
    """The facade of the wall a rake lies on. Two rakes meeting at an apex: the facade their feet's
    joining line runs toward. One rake: the facade other than that of the eave at its foot."""
    if len(camera.facades) <= 1:
        return camera.facades[0] if camera.facades else "front"
    ends = [rake.points[0], rake.points[-1]]
    for foot, other_foot in _rake_pairs(rake, lines, reach):
        found = camera.facade_of(foot, other_foot, tolerance)
        if found:
            return found
    eave_facades = {e.side for e in lines if e.edge in ("eave", "junction") and e.side
                    and any(_ends_meet(end, tip, reach) for end in ends for tip in (e.points[0], e.points[-1]))}
    others = [f for f in camera.facades if f not in eave_facades]
    return others[0] if len(eave_facades) == 1 and len(others) == 1 else None


def _feet(inches: float) -> str:
    return f"{inches / 12:.0f} ft" if inches >= 24 else f"{inches:.0f} in"


def _wall_reading(camera: Camera, solution: "Solution", placed: dict, by_id: dict, line: Line) -> float | None:
    """What a line measures if it lies on the nearest wall (the wall of the placed line closest to
    it across the photo): only a rough reading for a sloped line, enough to catch a slip."""
    xs = [p[0] for p in line.points]
    best = None
    for other_id, done in placed.items():
        if other_id == line.id or done.wall not in solution.walls or other_id not in by_id:
            continue
        theirs = [p[0] for p in by_id[other_id].points]
        gap = max(0.0, min(theirs) - max(xs), min(xs) - max(theirs))
        if best is None or gap < best[0]:
            best = (gap, done.wall)
    if best is None:
        return None
    points = [camera.on_wall(p, camera.wall_normal(best[1]), solution.walls[best[1]]["distance_in"]) for p in line.points]
    return None if any(p is None for p in points) else _length(points)


def _questioned(typed: float | None, suggested: float | None) -> str | None:
    """A sentence when a typed length is far from what the photo suggests; None otherwise."""
    if not typed or not suggested or 1 / TYPED_LENGTH_RATIO <= typed / suggested <= TYPED_LENGTH_RATIO:
        return None
    return (f"{_feet(typed)} seems {'short' if typed < suggested else 'long'} for this line; "
            f"the photo suggests about {_feet(suggested)}")


def solve(camera: Camera, lines: list[Line], measurement: dict | None, interpreted: dict | None = None,
          tolerance_deg: float = 3.0, overrides: dict | None = None) -> Solution:
    """Scale and 3-D for every line. `measurement` is Photo.scale's {facade, points or region_id,
    length_in}; `overrides` is {line id: {"mid": [x, y], "length_in"}} for lines the geometry
    cannot place."""
    solution = Solution()
    overrides = overrides or {}
    measured_facade = None
    reference_id = (measurement or {}).get("region_id")
    # A level line detection left without a side (its facade's vanishing point was found only by
    # the camera, services/camera._second_facade) takes the facade it runs toward. Still grouping
    # by vanishing point, never by slope.
    lines = [Line(line.id, line.points, line.edge,
                  line.side or (camera.facade_of(line.points[0], line.points[-1], tolerance_deg)
                                if line.edge in LEVEL else None))
             for line in lines]
    by_id = {line.id: line for line in lines}
    reach = CORNER_REACH_FRACTION * max(camera.width, camera.height)
    facades = camera.facades or ["front"]
    normals = {f: camera.wall_normal(f) for f in facades}

    if not measurement:
        solution.needs = "measurement"
    else:
        facade = measurement.get("facade") or facades[0]
        normal = normals.get(facade)
        if measurement.get("region_id"):
            reference = by_id.get(measurement["region_id"])
            pixels = reference.points if reference else None
        else:
            pixels = measurement.get("points")
        # The wall stands in front of the camera: its offset takes the sign that puts the measured
        # points there. (The wrong sign mirrors the house through the lens: lengths survive, but
        # up and down swap, and every rule asking which end of a line is the top goes wrong.)
        sign = 1.0
        if normal is not None and pixels:
            sign = 1.0 if float(normal @ camera.ray(pixels[0])) > 0 else -1.0
        unit = [_on_plane(camera, p, normal, sign) for p in pixels] if normal is not None and pixels else None
        if unit is None or any(p is None for p in unit) or _length(unit) <= 0:
            solution.needs = "measurement_wall"
        else:
            solution.walls[facade] = {"distance_in": sign * float(measurement["length_in"]) / _length(unit),
                                      "source": "measured"}
            measured_facade = facade

    # Each line's wall: its facade's for eaves and junctions, its gable's for rakes. A rake that reads
    # as a hip (no peak, eave end to ridge end) is placed by the hip rule instead, never on a wall.
    as_hip = {line.id for line in lines if line.edge == "rake" and line.id != reference_id
              and _reads_as_hip(line, lines, reach)}
    wall_of: dict[str, str] = {}
    for line in lines:
        if line.edge in ("eave", "junction") and line.side in normals:
            wall_of[line.id] = line.side
        elif line.edge == "rake" and line.id not in as_hip:
            gable = _gable_facade(camera, line, lines, reach, tolerance_deg)
            if gable in normals:
                wall_of[line.id] = gable
    # The line the salesperson measured is on the wall they measured it against, whatever its own
    # rule would say: the scale was taken from it there, so it reads exactly the length they typed.
    if reference_id and measured_facade:
        wall_of[reference_id] = measured_facade

    # Inheritance through shared corners, until nothing new is reached.
    if solution.walls:
        changed = True
        while changed:
            changed = False
            candidates: dict[str, list[float]] = {}
            for a in lines:
                known = wall_of.get(a.id)
                if known not in solution.walls:
                    continue
                for b in lines:
                    other = wall_of.get(b.id)
                    if other is None or other in solution.walls or normals.get(other) is None:
                        continue
                    for end in (a.points[0], a.points[-1]):
                        for tip in (b.points[0], b.points[-1]):
                            if _ends_meet(end, tip, reach):
                                corner = [(end[0] + tip[0]) / 2, (end[1] + tip[1]) / 2]
                                point = _on_plane(camera, corner, normals[known], solution.walls[known]["distance_in"])
                                if point is not None:
                                    candidates.setdefault(other, []).append(float(normals[other] @ point))
            for other, values in candidates.items():
                # Every point of a wall in view lies on one side of the camera. Corners that put the
                # wall through the camera, or on both sides of it, are corners of some other wall
                # (a misread line), and a median of them is a plane through the lens.
                values = [v for v in values if abs(v) > 1.0]
                if not values or (min(values) < 0 < max(values)):
                    continue
                solution.walls[other] = {"distance_in": float(np.median(values)), "source": "corner"}
                changed = True

    focal_note = None if camera.focal_source != "assumed" else "the lens is assumed, so depth is approximate"
    measured = next((f for f, w in solution.walls.items() if w["source"] == "measured"), None)

    def on_wall(line: Line, facade: str, method: str, plane: str) -> Placed:
        normal, distance = normals[facade], solution.walls[facade]["distance_in"]
        points = [_on_plane(camera, p, normal, distance) for p in line.points]
        if any(p is None for p in points):
            return Placed(line.id, needs="edge_on")
        inherited = facade != measured
        return Placed(line.id, points, _length(points), plane, method,
                      assumption=focal_note if inherited else None,
                      confidence=ASSUMED if inherited and focal_note else EXACT, wall=facade)

    placed: dict[str, Placed] = {}
    for line in lines:
        facade = wall_of.get(line.id)
        if facade is None:
            continue
        if facade not in solution.walls:
            placed[line.id] = Placed(line.id, needs="measurement")
            continue
        if line.id == reference_id:
            placed[line.id] = on_wall(line, facade, "measured", f"{facade} wall, your measurement")
            continue
        kind = "gable wall" if line.edge == "rake" else "wall"
        placed[line.id] = on_wall(line, facade, "gable" if line.edge == "rake" else "wall", f"{facade} {kind}")

    up = world_up(camera)
    rise, rise_source = pitch_hint(interpreted)

    def anchored_ends() -> list[tuple[list[float], np.ndarray, str, str]]:
        """Placed 3-D points a ridge can hang from: (pixel, point, what it is, how sure). Only a
        gable peak, two rakes meeting on their gable wall, is exact; a lone rake's top is as sure
        as the wall it was put on, and a hip's top is as sure as the hip."""
        out = []
        for line in lines:
            done = placed.get(line.id)
            if not done or not done.points3d or done.method == "override":
                continue
            top = max(range(len(done.points3d)), key=lambda i: float(done.points3d[i] @ up))
            if line.edge == "rake" and done.method in ("gable", "measured", "wall"):
                # A peak: another rake, of any size, comes down from this top.
                if any(other is not line and other.edge == "rake" and
                       any(_ends_meet(line.points[top], tip, reach) for tip in (other.points[0], other.points[-1]))
                       for other in lines):
                    out.append((line.points[top], done.points3d[top], "gable peak",
                                EXACT if not focal_note else STATED))
                else:
                    out.append((line.points[top], done.points3d[top], "rake top", STATED))
            elif line.edge == "hip" or done.method == "hip":
                out.append((line.points[top], done.points3d[top], "hip top", done.confidence or STATED))
        return out

    # (b) hips from eave corners, and rakes that read as hips.
    eave_ends = []
    for line in lines:
        done = placed.get(line.id)
        if line.edge in ("eave", "junction") and done and done.points3d and len(done.points3d) >= 2:
            level = _unit(done.points3d[-1] - done.points3d[0])
            eave_ends += [(line.points[0], done.points3d[0], level), (line.points[-1], done.points3d[-1], level)]
    for line in lines:
        if (line.edge != "hip" and line.id not in as_hip) or line.id in placed:
            continue
        for i, end in enumerate((line.points[0], line.points[-1])):
            corner = next(((pix, point, level) for pix, point, level in eave_ends if _ends_meet(end, pix, reach)), None)
            if corner is None:
                continue
            _, base, level = corner
            across = _unit(np.cross(up, level))
            top_pixel = line.points[-1] if i == 0 else line.points[0]
            toward = np.array(top_pixel, float) - np.array(end, float)
            best = None
            for s1 in (1, -1):
                for s2 in (1, -1):
                    bisector = _unit(s1 * level + s2 * across)
                    step = camera.project(base + bisector * 12.0) - camera.project(base)
                    score = float(_unit(step) @ _unit(toward))
                    if best is None or score > best[0]:
                        best = (score, bisector)
            normal = _unit(np.cross(best[1], up))
            top = _on_plane(camera, top_pixel, normal, float(normal @ base))
            if top is None:
                continue
            run = float(np.linalg.norm((top - base) - ((top - base) @ up) * up)) / np.sqrt(2)
            implied = 12.0 * float((top - base) @ up) / run if run > 1e-9 else float("inf")
            points = [base, top] if i == 0 else [top, base]
            confidence = STATED if IMPLAUSIBLE_PITCH[0] <= implied <= IMPLAUSIBLE_PITCH[1] else ASSUMED
            if line.id in as_hip and confidence == ASSUMED:
                break   # no believable hip either: it stays a rake, below
            placed[line.id] = Placed(
                line.id, points, _length(points), "over the corner bisector", "hip",
                assumption=("marked rake, but no second rake meets it at a peak, so it is read as a hip; "
                            "draw the other rake if this is a gable. " if line.id in as_hip else "")
                + f"adjacent roof faces share one pitch (implies {implied:.1f}/12)"
                + (f"; {focal_note}" if focal_note else ""),
                confidence=confidence)
            break

    # A rake read as a hip that the hip rule could not place (no placed eave at its foot) goes
    # back to the rake rule: its gable wall.
    for line in lines:
        if line.id not in as_hip or line.id in placed:
            continue
        gable = _gable_facade(camera, line, lines, reach, tolerance_deg)
        if gable not in normals:
            continue
        if gable not in solution.walls:
            # Its wall was never reached, since it was not a rake then: through its foot's corner.
            values = []
            for end in (line.points[0], line.points[-1]):
                for eave, done in placed.items():
                    if eave in by_id and done.wall in solution.walls and by_id[eave].edge in ("eave", "junction"):
                        tips = (by_id[eave].points[0], by_id[eave].points[-1])
                        for tip in tips:
                            if _ends_meet(end, tip, reach):
                                corner = [(end[0] + tip[0]) / 2, (end[1] + tip[1]) / 2]
                                point = _on_plane(camera, corner, normals[done.wall],
                                                  solution.walls[done.wall]["distance_in"])
                                if point is not None and abs(float(normals[gable] @ point)) > 1.0:
                                    values.append(float(normals[gable] @ point))
            if not values or min(values) < 0 < max(values):
                continue
            solution.walls[gable] = {"distance_in": float(np.median(values)), "source": "corner"}
        wall_of[line.id] = gable
        placed[line.id] = on_wall(line, gable, "gable", f"{gable} gable wall")

    # (a) ridges from a gable peak or a hip top. Anchored at both ends: the two anchors, so a small
    # move of either end cannot swap which one decides. At one end: level from it toward the
    # ridge's facade's vanishing point. Exact only from gable peaks; from a hip, as sure as the hip.
    anchors = anchored_ends()
    rank = {EXACT: 0, STATED: 1, ASSUMED: 2}

    def anchor_at(pixel):
        found = [a for a in anchors if _ends_meet(pixel, a[0], reach)]
        return min(found, key=lambda a: np.hypot(a[0][0] - pixel[0], a[0][1] - pixel[1])) if found else None

    for line in lines:
        if line.edge != "ridge" or line.id in placed:
            continue
        ends = (anchor_at(line.points[0]), anchor_at(line.points[-1]))
        if ends[0] is not None and ends[1] is not None and ends[0][1] is not ends[1][1]:
            points = [ends[0][1], ends[1][1]]
            whats = sorted({ends[0][2], ends[1][2]})
            confidence = max(ends[0][3], ends[1][3], key=rank.get)
            assumption = None if whats == ["gable peak"] else "from a hip, which assumes one pitch"
            notes = [n for n in (assumption, focal_note) if n]
            placed[line.id] = Placed(line.id, points, _length(points), f"between the {' and the '.join(whats)}s"
                                     if len(whats) == 1 else f"between the {whats[0]} and the {whats[1]}", "ridge",
                                     assumption="; ".join(notes) or None, confidence=confidence)
            continue
        if line.side not in normals or not camera.horizontal:
            continue
        level = camera.direction(camera.horizontal[camera.facades.index(line.side)])
        for i in (0, 1):
            anchor = ends[i]
            if anchor is None:
                continue
            _, start, what, sure = anchor
            other = line.points[-1] if i == 0 else line.points[0]
            finish = _closest_on_line(camera, other, start, level)
            points = [start, finish] if i == 0 else [finish, start]
            assumption = {"gable peak": None, "rake top": "a lone rake's wall, no gable peak in view"}.get(
                what, "from a hip, which assumes one pitch")
            notes = [n for n in (assumption, focal_note) if n]
            placed[line.id] = Placed(line.id, points, _length(points), f"level from the {what}", "ridge",
                                     assumption="; ".join(notes) or None, confidence=sure)
            break

    # (c) the rest of the ridges, hips and valleys: the face rising from the nearest eave.
    eaves = [(line, placed[line.id]) for line in lines
             if line.edge == "eave" and line.id in placed and placed[line.id].points3d]
    angle = float(np.arctan2(rise, 12.0))
    for line in lines:
        if line.id in placed or line.edge not in ("ridge", "hip", "valley"):
            continue
        if not eaves:
            placed[line.id] = Placed(line.id, needs="measurement" if not solution.walls else "anchor")
            continue
        middle = np.mean(np.asarray(line.points, float), axis=0)

        def distance(item) -> float:
            a, b = np.asarray(item[0].points[0], float), np.asarray(item[0].points[-1], float)
            t = np.clip(float((middle - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-9), 0, 1)
            return float(np.linalg.norm(middle - (a + t * (b - a))))

        _, eave = min(eaves, key=distance)
        level = _unit(eave.points3d[-1] - eave.points3d[0])
        inward = _unit(np.cross(up, level))
        centre = (eave.points3d[0] + eave.points3d[-1]) / 2
        if float(inward @ centre) < 0:
            inward = -inward    # the roof rises away from the camera, into the house
        normal = _unit(np.cross(level, np.cos(angle) * inward + np.sin(angle) * up))
        points = [_on_plane(camera, p, normal, float(normal @ centre)) for p in line.points]
        if any(p is None for p in points):
            placed[line.id] = Placed(line.id, needs="edge_on")
            continue
        placed[line.id] = Placed(line.id, points, _length(points), "roof face from the nearest eave", "pitch",
                                 assumption=f"depth from an {rise_source} {rise:g}/12 pitch"
                                 + (f"; {focal_note}" if focal_note else ""),
                                 confidence=ASSUMED)

    # Anything left, and every line the salesperson overrode.
    for line in lines:
        # Only a midpoint or a length is a line override; other keys belong to areas.
        override = {k: v for k, v in (overrides.get(line.id) or {}).items() if k in ("mid", "length_in")}
        done = placed.get(line.id)
        if override:
            length = override.get("length_in") or (done.length_in if done and done.points3d else None)
            placed[line.id] = Placed(line.id, done.points3d if done else None, length, "set by hand", "override",
                                     assumption="midpoint placed by hand",
                                     confidence=STATED if length else None,
                                     needs=None if length else "length",
                                     warning=_questioned(override.get("length_in"),
                                                         done.length_in if done and done.points3d
                                                         else _wall_reading(camera, solution, placed, by_id, line)))
        elif done is None:
            reason = "measurement" if not solution.walls else ("side" if line.side is None else "override")
            placed[line.id] = Placed(line.id, needs=reason)
    solution.lines = placed
    if measured_facade:
        solution.warning = _implausible_wall(camera, measured_facade, lines, placed)
    return solution


def _implausible_wall(camera: Camera, facade: str, lines: list[Line], placed: dict) -> str | None:
    """A sentence when the measured wall's eaves span wider than a house; None otherwise. The span
    is along the wall, end to end over every eave on it, so a broken eave counts whole."""
    points = [point for line in lines if line.edge in ("eave", "junction")
              for done in [placed.get(line.id)] if done and done.wall == facade and done.points3d
              and done.method in ("wall", "measured") for point in done.points3d]
    if len(points) < 2 or not camera.horizontal or facade not in camera.facades:
        return None
    level = _unit(camera.direction(camera.horizontal[camera.facades.index(facade)]))
    along = [float(point @ level) for point in points]
    feet = (max(along) - min(along)) / 12.0
    if feet <= PLAUSIBLE_WALL_FT:
        return None
    return (f"This measurement makes the {facade} of the house {feet:.0f} ft wide, which is "
            f"more than a house usually is. Check the length you typed, "
            f"or measure something of a known size instead: a single garage door is about 9 ft, "
            f"a double 16 ft, an entry door 80 in tall.")
