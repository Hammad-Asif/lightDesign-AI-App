"""Which wall a level line, or a tapped reference, is on (M3: scale).

Direction alone is weak near eye level. Every horizontal vanishing point lies on the horizon, so
a level line at camera height points at all of them: about 2 degrees of margin per foot of height
above the camera, at house distances. A garage-door top is such a line, and a reference put on
the wrong wall scales every length wrong. So a line's facade is decided in this order:

  1. direction, but only with a clear margin (MARGIN_MIN_DEG) between the nearest vanishing
     point and the next; a side given by detection or the salesperson stands when the
     direction allows it;
  2. structure, for the rest: a line continuing a placed one end to end (collinear in the photo)
     is on its wall; one meeting it at an angle there is on the other wall (the wall turns); a
     line under or over a placed level line, overlapping it across the photo, shares its wall;
  3. still undecided: ambiguous, with its candidate facades. scale.solve tries each and keeps
     the choice under which the house is consistent (services/checks.py).

Set-back walls. A junction (roof meeting a wall) with an eave of the same facade below it is the
foot of a wall standing on that roof: a dormer's face, or the upper storey behind a porch roof.
It faces the same way as the main wall but sits deeper by the roof between. The junction and the
level lines above it, across its span, are on that set-back wall; scale.solve links its distance
to the main wall's through the roof face (pitch) between the eave and the junction. A dormer whose
foot is hidden is known by its eave alone: an eave standing above a longer eave of its facade.
"""

from dataclasses import dataclass, field

import numpy as np

from app.services.camera import Camera

MARGIN_MIN_DEG = 4.0      # nearest vanishing point must beat the next by this to decide alone
COLLINEAR_DEG = 3.0       # two pieces meeting end to end this close in direction are one wall
TURN_DEG = 10.0           # ... this far apart, the wall turns there
OVERLAP_SHARE = 0.5       # a line under or over another this much across shares its wall
# A line (or a chain of collinear pieces) shorter than this many corner reaches has a direction
# worth little: its side comes from its neighbours, and it decides no other line's.
MIN_LINE_REACHES = 1.5
# A line whose direction fits no wall may borrow a sure neighbour's side only when it is still
# within this of that wall's direction (a sloped line labelled "eave" is refused).
BORROW_MAX_DEG = 10.0
SURE = ("direction", "given", "only")
LEVEL = ("eave", "junction", "ridge")
WALL_EDGES = ("eave", "junction")


@dataclass
class Assignment:
    facade: str | None
    how: str                                   # given | direction | only | chain | turn | above | below | on_line | ambiguous | none
    margin: float | None = None                # degrees between the nearest vanishing point and the next
    candidates: list[str] = field(default_factory=list)
    via: str | None = None                     # the line that decided it, for structure

    def to_dict(self) -> dict:
        return {"facade": self.facade, "how": self.how,
                "margin_deg": round(self.margin, 2) if self.margin is not None and np.isfinite(self.margin) else None,
                "candidates": self.candidates, "via": self.via}


@dataclass
class SetBack:
    key: str                  # the wall's key in Solution.walls, e.g. "front+1"
    facade: str
    junction: str             # line id of its foot
    eave: str                 # the eave below whose roof face rises to the junction
    members: set = field(default_factory=set)
    span: tuple = (0.0, 0.0)  # x range of the junction, px

    def to_dict(self) -> dict:
        return {"key": self.key, "facade": self.facade, "junction": self.junction, "eave": self.eave,
                "members": sorted(self.members)}


def _ends(line) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(line.points[0], float), np.asarray(line.points[-1], float)


def _span(points) -> tuple[float, float]:
    xs = [float(p[0]) for p in points]
    return min(xs), max(xs)


def _overlap(a: tuple, b: tuple) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _y_at(points, x: float) -> float:
    """The polyline's height in the photo at x (clamped to its ends)."""
    pts = sorted(((float(p[0]), float(p[1])) for p in points))
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1e-9)
    return pts[-1][1]


def _direction_deg(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _angle_between(u: float, v: float) -> float:
    """Undirected angle between two image directions in degrees, in [0, 90]."""
    d = abs((u - v) % 180.0)
    return min(d, 180.0 - d)


def by_direction(camera: Camera, a, b, facades: list[str], tolerance: float, given: str | None) -> Assignment:
    """The facade the segment a→b runs toward, only when the margin is clear."""
    angles = [(angle, f) for angle, f in camera.facade_angles(a, b) if f in facades]
    if not angles:
        # No horizontal vanishing point: every wall faces the camera.
        only = facades[0] if facades else None
        return Assignment(only, "only" if only else "none", None, [only] if only else [])
    best_angle, best = angles[0]
    if len(angles) == 1:
        if best_angle <= tolerance:
            return Assignment(best, "direction", float("inf"), [best])
        return Assignment(given, "given", None, [given]) if given in facades else Assignment(None, "none")
    margin = angles[1][0] - best_angle
    within = [f for angle, f in angles if angle <= tolerance + MARGIN_MIN_DEG]
    if best_angle <= tolerance and margin >= MARGIN_MIN_DEG:
        given_angle = next((angle for angle, f in angles if f == given), None)
        if given in facades and given != best and given_angle is not None and given_angle <= tolerance:
            return Assignment(given, "given", margin, [given])
        return Assignment(best, "direction", margin, [best])
    if not within:
        return Assignment(given, "given", margin, [given]) if given in facades else Assignment(None, "none", margin)
    if len(within) == 1:
        if best_angle <= tolerance:
            return Assignment(within[0], "direction", margin, within)
        return Assignment(given, "given", margin, [given]) if given in facades else Assignment(None, "none", margin)
    # Put a given side first among the candidates: the preference when nothing else decides.
    if given in within:
        within = [given] + [f for f in within if f != given]
    return Assignment(None, "ambiguous", margin, within)


def _span_px(line) -> float:
    a, b = _ends(line)
    return float(np.linalg.norm(b - a))


def _chained(a, b, reach: float) -> bool:
    """Two lines continuing each other end to end, collinear in the photo."""
    a0, a1 = _ends(a)
    b0, b1 = _ends(b)
    if min(np.linalg.norm(p - q) for p in (a0, a1) for q in (b0, b1)) > reach:
        return False
    return _angle_between(_direction_deg(a0, a1), _direction_deg(b0, b1)) <= COLLINEAR_DEG


def _groups(members: list, joined) -> list[list]:
    """Connected groups of members under the pairwise relation `joined`."""
    parent = {id(m): m for m in members}
    root = {id(m): id(m) for m in members}

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if joined(a, b):
                root[find(id(a))] = find(id(b))
    out: dict[int, list] = {}
    for m in members:
        out.setdefault(find(id(m)), []).append(parent[id(m)])
    return list(out.values())


def _extremes(group: list) -> tuple[np.ndarray, np.ndarray]:
    """The two farthest-apart ends of a chain of lines: its direction over its whole length."""
    ends = [p for line in group for p in _ends(line)]
    best = (ends[0], ends[-1])
    for i, p in enumerate(ends):
        for q in ends[i + 1:]:
            if np.linalg.norm(q - p) > np.linalg.norm(best[1] - best[0]):
                best = (p, q)
    return best


def assign_lines(camera: Camera, lines: list, facades: list[str], tolerance: float, reach: float) -> dict[str, Assignment]:
    """Every level line's facade: direction with a clear margin, then structure.

    Pieces of one line broken by detection (collinear, end to end) are read together, over their
    whole length: a 30 px piece's direction is noise, the 300 px line it belongs to is not. A line
    (or chain) still too short has no direction of its own ("short"): its side comes from its
    neighbours. A line whose direction fits no wall ("none") may borrow a sure neighbour's side
    when it is within BORROW_MAX_DEG of that wall's direction."""
    out: dict[str, Assignment] = {}
    level = [line for line in lines if line.edge in LEVEL and len(line.points) >= 2]
    min_len = MIN_LINE_REACHES * reach
    for group in _groups(level, lambda x, y: x.edge == y.edge and _chained(x, y, reach)):
        a, b = _extremes(group)
        for line in group:
            if float(np.linalg.norm(b - a)) < min_len:
                given = line.side
                out[line.id] = (Assignment(given, "given", None, [given]) if given in facades
                                else Assignment(None, "short"))
            else:
                out[line.id] = by_direction(camera, a, b, facades, tolerance, line.side)

    changed = True
    while changed:
        changed = False
        for line in level:
            mine = out[line.id]
            if mine.how == "ambiguous":
                found = _by_structure(line, level, out, reach, min_len)
                if found is not None and found[0] in mine.candidates:
                    out[line.id] = Assignment(found[0], found[1], mine.margin, mine.candidates, found[2])
                    changed = True
            elif mine.how in ("none", "short"):
                found = _by_structure(line, level, out, reach, min_len, sure_only=True)
                if found is None:
                    continue
                if mine.how == "none":
                    angles = {f: angle for angle, f in camera.facade_angles(*_ends(line))}
                    if angles.get(found[0], 90.0) > BORROW_MAX_DEG:
                        continue
                out[line.id] = Assignment(found[0], "borrowed", mine.margin, [found[0]], found[2])
                changed = True
    return out


def _decided(assignment: Assignment) -> bool:
    return assignment.facade is not None and assignment.how not in ("ambiguous", "none", "short")


def _by_structure(line, level: list, out: dict, reach: float, min_len: float = 0.0,
                  sure_only: bool = False) -> tuple[str, str, str] | None:
    """The side a line takes from a decided neighbour long enough to trust (`sure_only`: one whose
    own direction or a given side decided it, not one that borrowed its side in turn)."""
    a, b = _ends(line)
    mine = _direction_deg(a, b)

    def trusted(other, theirs) -> bool:
        return (theirs is not None and _decided(theirs) and _span_px(other) >= min_len
                and (not sure_only or theirs.how in SURE))

    # 1. End to end with a decided line: collinear continues its wall, an angle turns it.
    for other in level:
        theirs = out.get(other.id)
        if other is line or not trusted(other, theirs):
            continue
        c, d = _ends(other)
        if min(np.linalg.norm(p - q) for p in (a, b) for q in (c, d)) > reach:
            continue
        between = _angle_between(mine, _direction_deg(c, d))
        if between <= COLLINEAR_DEG:
            return theirs.facade, "chain", other.id
        if between >= TURN_DEG:
            others = [f for f in out[line.id].candidates if f != theirs.facade]
            if len(others) == 1:
                return others[0], "turn", other.id
    # 2. Under or over a decided wall line, across most of this one's width: the nearest such.
    span = _span(line.points)
    width = max(span[1] - span[0], 1.0)
    middle_y = float(np.mean([p[1] for p in line.points]))
    best = None
    for other in level:
        theirs = out.get(other.id)
        if other is line or other.edge not in WALL_EDGES or not trusted(other, theirs):
            continue
        share = _overlap(span, _span(other.points)) / width
        if share < OVERLAP_SHARE:
            continue
        gap = _y_at(other.points, (span[0] + span[1]) / 2) - middle_y
        how = "above" if gap < 0 else "below"
        # Prefer a line above: the higher a line, the clearer its direction.
        rank = (0 if how == "above" else 1, abs(gap))
        if best is None or rank < best[0]:
            best = (rank, theirs.facade, how, other.id)
    if best is not None:
        return best[1], best[2], best[3]
    return None


def assign_points(camera: Camera, points, lines: list, assignment: dict[str, Assignment], facades: list[str],
                  tolerance: float, reach: float, given: str | None = None) -> Assignment:
    """The facade of a tapped segment (the reference, a door's width): given, on a line, by
    direction with a clear margin, else the decided wall line nearest above it."""
    if given in facades:
        return Assignment(given, "given", None, [given])
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    # On a level line: that line's wall.
    for line in lines:
        theirs = assignment.get(line.id)
        if line.edge in WALL_EDGES and theirs is not None and _decided(theirs) and \
                all(_to_polyline(p, line.points) <= reach for p in (a, b)):
            return Assignment(theirs.facade, "on_line", None, [theirs.facade], line.id)
    found = by_direction(camera, a, b, facades, tolerance, None)
    if _decided(found):
        return found
    x, y = float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
    best = None
    for line in lines:
        theirs = assignment.get(line.id)
        if line.edge not in WALL_EDGES or theirs is None or not _decided(theirs):
            continue
        x0, x1 = _span(line.points)
        if not x0 - reach <= x <= x1 + reach:
            continue
        gap = y - _y_at(line.points, x)
        if gap <= 0:
            continue
        if best is None or gap < best[0]:
            best = (gap, theirs.facade, line.id)
    if best is not None and (not found.candidates or best[1] in found.candidates):
        return Assignment(best[1], "above", found.margin, found.candidates or [best[1]], best[2])
    if found.how == "ambiguous":
        return found
    return found if found.facade else Assignment(None, "none", found.margin, list(facades))


def _to_polyline(point, polyline) -> float:
    p = np.asarray(point, float)
    best = float("inf")
    for a, b in zip(polyline[:-1], polyline[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = np.clip(float((p - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-9), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def setbacks(lines: list, facade_of: dict[str, str | None], reach: float) -> list[SetBack]:
    """Walls standing on a roof: each junction with an eave of its facade below it."""
    found: list[SetBack] = []
    wall_lines = [line for line in lines if line.edge in WALL_EDGES and facade_of.get(line.id)]
    for junction in (line for line in wall_lines if line.edge == "junction"):
        facade = facade_of[junction.id]
        span = _span(junction.points)
        width = max(span[1] - span[0], 1.0)
        middle = (span[0] + span[1]) / 2
        foot_y = _y_at(junction.points, middle)
        below = None
        for eave in wall_lines:
            if eave.edge != "eave" or facade_of.get(eave.id) != facade:
                continue
            if _overlap(span, _span(eave.points)) / width < 0.3:
                continue
            gap = _y_at(eave.points, middle) - foot_y
            if gap > reach and (below is None or gap < below[0]):
                below = (gap, eave.id)
        if below is None:
            continue
        found.append(SetBack(key=f"{facade}+{len(found) + 1}", facade=facade, junction=junction.id,
                             eave=below[1], members={junction.id}, span=span))
    found += _dormer_eaves(wall_lines, facade_of, reach, found)
    if not found:
        return found
    # Each level wall line above a junction, across most of its width, belongs to the nearest
    # such junction below it.
    for line in wall_lines:
        if any(s.junction == line.id for s in found):
            continue
        span = _span(line.points)
        width = max(span[1] - span[0], 1.0)
        line_y = float(np.mean([p[1] for p in line.points]))
        best = None
        for setback in found:
            if facade_of.get(line.id) != setback.facade:
                continue
            junction_width = max(setback.span[1] - setback.span[0], 1.0)
            share = _overlap(span, setback.span) / min(width, junction_width)
            if share < 0.8 or width > 1.5 * junction_width + 2 * reach:
                continue
            junction = next(l for l in wall_lines if l.id == setback.junction)
            gap = _y_at(junction.points, (span[0] + span[1]) / 2) - line_y
            if gap > 0 and (best is None or gap < best[0]):
                best = (gap, setback)
        if best is not None:
            best[1].members.add(line.id)
    return found


def _dormer_eaves(wall_lines: list, facade_of: dict, reach: float, found: list[SetBack]) -> list[SetBack]:
    """Set-back walls known only by their eave: an eave of a facade standing above a longer eave of
    the same facade, across most of its own width, with no junction seen at its foot (a dormer
    whose foot is hidden behind the roof). The eave stands in for the foot: scale.solve links its
    wall through the roof face rising from the eave below, at the eave's own pixels. That puts it
    where its line of sight meets the roof, a little deeper than the dormer's face, and its own
    overhang brings it back toward the face."""
    taken = {member for setback in found for member in setback.members}
    eaves = [line for line in wall_lines if line.edge == "eave" and line.id not in taken]
    candidates: list[tuple[object, str]] = []
    for eave in eaves:
        facade = facade_of[eave.id]
        span = _span(eave.points)
        width = max(span[1] - span[0], 1.0)
        middle = (span[0] + span[1]) / 2
        y = _y_at(eave.points, middle)
        below = None
        for other in eaves:
            if other is eave or facade_of.get(other.id) != facade:
                continue
            other_span = _span(other.points)
            if other_span[1] - other_span[0] < width or _overlap(span, other_span) / width < OVERLAP_SHARE:
                continue
            gap = _y_at(other.points, middle) - y
            if gap > reach and (below is None or gap < below[0]):
                below = (gap, other.id)
        if below is not None:
            candidates.append((eave, below[1]))
    # The eave below must be on the main wall: a dormer above a dormer is not linked here.
    dormers = {eave.id for eave, _ in candidates}
    out: list[SetBack] = []
    for eave, below in candidates:
        if below in dormers:
            continue
        facade = facade_of[eave.id]
        out.append(SetBack(key=f"{facade}+{len(found) + len(out) + 1}", facade=facade, junction=eave.id,
                           eave=below, members={eave.id}, span=_span(eave.points)))
    return out


def physical_walls(lines: list, key_of: dict[str, str], facades: list[str], reach: float) -> tuple[dict, dict]:
    """A facade's wall lines grouped into the walls they lie on. Lines continuing each other end to
    end, collinear in the photo, are one wall; separate runs facing the same way are separate walls
    (a garage wing's face and the entry's, an L's two arms), each with its own distance. The
    longest run keeps the facade's key, the others are "front.2", "front.3"... by length.
    Returns (key_of with the new keys, {new key: facade})."""
    by_id = {line.id: line for line in lines}
    out = dict(key_of)
    split: dict[str, str] = {}
    for facade in facades:
        members = [by_id[line_id] for line_id, key in key_of.items() if key == facade and line_id in by_id]
        if len(members) < 2:
            continue
        groups = sorted(_groups(members, lambda a, b: _chained(a, b, reach)),
                        key=lambda group: -sum(_span_px(line) for line in group))
        for n, group in enumerate(groups[1:], start=2):
            key = f"{facade}.{n}"
            split[key] = facade
            for line in group:
                out[line.id] = key
    return out, split


def facade_of_key(key: str) -> str:
    """The facade a wall key belongs to: "front.2", "front+1", "front#1", "front@1a2b3c4d" (a
    dormer's side wall) are all "front"."""
    for mark in (".", "+", "#", "@"):
        key = key.partition(mark)[0]
    return key


def nearest_wall_key(points, lines: list, key_of: dict[str, str], facade: str | None, reach: float) -> str | None:
    """The wall (key) of the facade's wall line nearest a tapped segment, across the photo within
    its span: which of a facade's walls the segment is on. None when no such line is near."""
    if facade is None:
        return None
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    x, y = float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
    best = None
    for line in lines:
        key = key_of.get(line.id)
        if key is None or line.edge not in WALL_EDGES or facade_of_key(key) != facade or any(m in key for m in "#+@"):
            continue    # set-back walls are found by setback_of_points; gable and dormer walls are not the main face
        span = _span(line.points)
        if not span[0] - reach <= x <= span[1] + reach:
            continue
        gap = abs(_y_at(line.points, x) - y)
        if best is None or gap < best[0]:
            best = (gap, key)
    return best[1] if best else None


def setback_of_points(points, lines: list, found: list[SetBack], facade: str | None, reach: float) -> SetBack | None:
    """The set-back wall a tapped segment is on: above its junction, within its span."""
    if not found or facade is None:
        return None
    a, b = np.asarray(points[0], float), np.asarray(points[-1], float)
    x, y = float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
    by_id = {line.id: line for line in lines}
    best = None
    for setback in found:
        if setback.facade != facade or not setback.span[0] - reach <= x <= setback.span[1] + reach:
            continue
        gap = _y_at(by_id[setback.junction].points, x) - y
        if gap > -reach and (best is None or gap < best[0]):
            best = (gap, setback)
    return best[1] if best else None
