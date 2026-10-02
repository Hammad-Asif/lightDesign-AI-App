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
    the facade other than its eave's.
  - ridge, hip, valley, by the first rule that applies, each stating its assumption:
      (a) a ridge ending at a gable apex (the top of a placed rake, which is on the gable wall)
          or at a hip top: exact. It runs level from there toward its facade's vanishing point.
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

    def to_dict(self) -> dict:
        return {"walls": self.walls, "lines": {k: v.to_dict() for k, v in self.lines.items()}, "needs": self.needs}


# -- small geometry ---------------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else v


def _length(points3d: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points3d[:-1], points3d[1:])))


def _on_plane(camera: Camera, pixel, normal: np.ndarray, offset: float) -> np.ndarray | None:
    return camera.on_wall(pixel, normal, offset)


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


def _gable_facade(camera: Camera, rake: Line, lines: list[Line], reach: float, tolerance: float) -> str | None:
    """The facade of the wall a rake lies on. Two rakes meeting at an apex: the facade their feet's
    joining line runs toward. One rake: the facade other than that of the eave at its foot."""
    if len(camera.facades) <= 1:
        return camera.facades[0] if camera.facades else "front"
    ends = [rake.points[0], rake.points[-1]]
    span = float(np.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]))
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
                    foot, other_foot = ends[1 - i], their_other
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
        unit = [_on_plane(camera, p, normal, 1.0) for p in pixels] if normal is not None and pixels else None
        if unit is None or any(p is None for p in unit) or _length(unit) <= 0:
            solution.needs = "measurement_wall"
        else:
            solution.walls[facade] = {"distance_in": float(measurement["length_in"]) / _length(unit),
                                      "source": "measured"}
            measured_facade = facade

    # Each line's wall: its facade's for eaves and junctions, its gable's for rakes.
    wall_of: dict[str, str] = {}
    for line in lines:
        if line.edge in ("eave", "junction") and line.side in normals:
            wall_of[line.id] = line.side
        elif line.edge == "rake":
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

    def anchored_ends() -> list[tuple[list[float], np.ndarray, str]]:
        """Placed 3-D points a ridge or hip can hang from: (pixel, point, what it is)."""
        out = []
        for line in lines:
            done = placed.get(line.id)
            if not done or not done.points3d:
                continue
            if line.edge == "rake":
                # The apex is the rake's top end: the higher of its two ends in the world.
                top = max(range(len(done.points3d)), key=lambda i: float(done.points3d[i] @ up))
                out.append((line.points[top], done.points3d[top], "apex"))
            elif line.edge == "hip":
                top = max(range(len(done.points3d)), key=lambda i: float(done.points3d[i] @ up))
                out.append((line.points[top], done.points3d[top], "hip top"))
        return out

    # (b) hips from eave corners.
    eave_ends = []
    for line in lines:
        done = placed.get(line.id)
        if line.edge in ("eave", "junction") and done and done.points3d and len(done.points3d) >= 2:
            level = _unit(done.points3d[-1] - done.points3d[0])
            eave_ends += [(line.points[0], done.points3d[0], level), (line.points[-1], done.points3d[-1], level)]
    for line in lines:
        if line.edge != "hip" or line.id in placed:
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
            placed[line.id] = Placed(
                line.id, points, _length(points), "over the corner bisector", "hip",
                assumption=f"adjacent roof faces share one pitch (implies {implied:.1f}/12)"
                + (f"; {focal_note}" if focal_note else ""),
                confidence=confidence)
            break

    # (a) ridges from an apex or a hip top, level toward their facade's vanishing point.
    anchors = anchored_ends()
    for line in lines:
        if line.edge != "ridge" or line.id in placed or line.side not in normals:
            continue
        level = camera.direction(camera.horizontal[camera.facades.index(line.side)]) if camera.horizontal else None
        for i, end in enumerate((line.points[0], line.points[-1])):
            anchor = next(((pix, point, what) for pix, point, what in anchors if _ends_meet(end, pix, reach)), None)
            if anchor is None or level is None:
                continue
            _, start, what = anchor
            other = line.points[-1] if i == 0 else line.points[0]
            finish = _closest_on_line(camera, other, start, level)
            points = [start, finish] if i == 0 else [finish, start]
            placed[line.id] = Placed(line.id, points, _length(points), f"level from the {what}", "ridge",
                                     assumption=focal_note if what == "apex" else
                                     "from a hip, which assumes one pitch" + (f"; {focal_note}" if focal_note else ""),
                                     confidence=EXACT if what == "apex" and not focal_note else STATED)
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
    return solution
