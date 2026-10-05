"""Real scale and 3-D position for every roof line of a photo (M3).

Everything here is in the camera's frame (services/camera.py), in inches: the salesperson's one
measurement fixes how far its wall is from the camera, and from there every line is placed in
3-D by a camera ray meeting a plane. What each line needed to assume is recorded with it.

Walls. A facade is a vertical plane n·X = d; camera.py gives n from the vanishing points. Which
facade a level line (or the reference) is on is decided by services/walls.py: direction only with a
clear margin, structure otherwise, because near eye level every horizontal vanishing point is in
the same direction. A line still undecided is tried on each candidate facade, and the choice under
which the house passes services/checks.py best is kept (its lines say "side chosen by
consistency"). A wall standing on a roof (a dormer's face, an upper storey behind a porch roof)
is a set-back wall of its facade, keyed "front+1": parallel to the main wall, deeper by the roof
between, and linked to it through that roof face at the pitch.

The measured wall's d makes the measurement come out at its real length. Another wall inherits d
through a corner it shares with a known one: where an eave or rake of each meet, the point is on
both walls, so it is fixed in 3-D by the known wall and the other wall's plane must pass through
it. Two walls of one facade never pass a distance through a corner: parallel planes share no
corner. Walls no corner or roof face reaches stay unknown ("needs a measurement").

Overhang. An eave hangs EAVE_OVERHANG_IN in front of its wall, so it is placed on a plane that
much nearer the camera; a junction sits on the wall itself. walls[...] always holds the wall's own
distance: a measurement along an eave (a picked eave line, or two points tapped on one) sets the
eave's plane, and the wall is the overhang behind it. Corners pass a distance from one line's plane
to the other's the same way.

Pitch. A rake lies in its gable wall, so its slope there is the roof's pitch whatever the scale:
the rakes on gable walls the camera sees well measure it before anything is placed. Else the
photo's estimated standard pitch, else 6/12. Every face is taken to share it.

Lines, by edge:
  - eave: on its wall's eave plane. junction: on its wall.
  - rake: on its gable wall when the camera sees that wall well (view rays meeting it at
    GRAZING_MIN or more), else on the roof face rising from the eave at its foot. The gable wall is
    the facade the line joining two rakes' feet runs toward; with one rake in view, the facade
    other than its eave's (inferred, so never exact). A rake is read as a hip when the roof shows
    on both its sides (Line.topology "shared"), when two eaves of different walls meet at its
    foot, or when, with no second rake at its top, it rises from an eave's end to a ridge's or
    hip's end; a hip reading that implies an implausible pitch goes back to the rake rule.
  - ridge, hip, valley, by the first rule that applies, each stating its assumption:
      (a) a ridge ending at a gable peak or a rake or hip top: between the two anchors, or level
          from one toward its facade's vanishing point. As sure as its anchors.
      (b) a hip rising from an eave corner: adjacent faces share one pitch, so the hip lies over
          the corner's bisector in plan: the sum of the directions of the eaves meeting there, away
          from the corner; with one eave, its direction plus the horizontal inward from it (away
          from the camera). Never from image directions: a flat step into the house runs down the
          photo when the camera is below the eave, and the outside bisector then matched the
          rising hip, halving every hip in a standing photo. The top is the camera ray meeting
          that vertical plane.
      (c) otherwise: the roof face rising from the nearest placed eave at the pitch.
  - anything else: the salesperson's override, a midpoint handle and a typed length
    (override_fraction), or "needs".
"""

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np

from app.services import camera as camera_module
from app.services import checks as checks_module
from app.services import walls as walls_module
from app.services.camera import Camera

CORNER_REACH_FRACTION = 0.02    # of the photo's long edge: line ends this close are one corner
DEFAULT_PITCH_RISE = 6.0        # per 12 of run, when the photo gives no estimate
EAVE_OVERHANG_IN = 18.0         # a typical eave overhang: how far an eave sits in front of its wall
PLAUSIBLE_PITCH = checks_module.PLAUSIBLE_PITCH   # rise per 12 a hip or rake may imply
LEVEL = {"eave", "junction", "ridge"}
WALL_EDGES = ("eave", "junction")
RAKE_PAIR_RATIO = 2.0           # two rakes this unequal in the photo are not one gable's two sides
TYPED_LENGTH_RATIO = 2.0        # a typed length this far off what the photo suggests is questioned
# A plane the view rays meet at less than this is seen nearly edge-on: a pixel's error becomes a
# large error in depth along it (a front-on gable wall: a 22 ft rake read as 41 ft).
GRAZING_MIN = float(np.sin(np.radians(8.0)))
MAX_CHOICES = 32                # wall choices tried for the undecided lines and the reference
REFERENCE = "__reference__"
# A line shorter than this share of the long edge (1.5 corner reaches) has a direction worth
# little: it does not vote on the pitch, pair into a gable, or decide another line's side.
MIN_LINE_REACHES = 1.5
# Rakes a gable's two sides may differ by, in the photo, before the plane test (which checks that
# both imply one pitch) accepts them as a pair.
PARTNER_RATIO = 3.0
# Per-line uncertainty: each length is solved again under the photo's own errors (the reference
# tapped a pixel off at each end, a vanishing point off by half a degree, the focal length off by
# its doubt, the pitch off by a step); a length moving more than SPREAD_MAX is not shown.
SPREAD_MAX = 0.15
VP_JITTER_DEG = 0.5
FOCAL_DOUBT = {"vp": 0.08, "exif": 0.05, "assumed": 0.12}
PITCH_DOUBT = {"measured": 1.0, "estimated": 2.0, "assumed": 2.0}
# Nothing of the house lies behind the camera, or beyond its farthest wall by more than this.
DEPTH_SLACK_IN = 720.0

EXACT, STATED, ASSUMED = "exact", "stated", "assumed"
RANK = {EXACT: 0, STATED: 1, ASSUMED: 2}


def _worst(*levels: str | None) -> str | None:
    known = [level for level in levels if level]
    return max(known, key=RANK.get) if known else None


@dataclass
class Line:
    id: str
    points: list[list[float]]   # photo pixels
    edge: str | None
    side: str | None
    topology: str | None = None  # "shared": roof on both sides (hip, valley, ridge); "free": one


@dataclass
class Placed:
    id: str
    points3d: list[np.ndarray] | None = None
    length_in: float | None = None
    plane: str | None = None        # which surface it was placed on, in words
    method: str = "none"            # measured | wall | gable | ridge | hip | pitch | valley | override | estimate | none
    assumption: str | None = None
    confidence: str | None = None   # exact | stated | assumed
    needs: str | None = None        # why it could not be placed
    wall: str | None = None         # the wall (key in Solution.walls) whose plane holds it
    warning: str | None = None      # a typed length the photo disagrees with
    side: str | None = None         # the facade a level line was given
    side_how: str | None = None     # how: walls.Assignment.how, or "chosen" by consistency
    spread: float | None = None     # relative: how far the length moves under the photo's own errors

    def to_dict(self) -> dict:
        return {"wall": self.wall, "warning": self.warning, "side": self.side, "side_how": self.side_how,
                "points3d": [[round(float(v), 2) for v in p] for p in self.points3d] if self.points3d else None,
                "length_in": round(self.length_in, 1) if self.length_in is not None else None,
                "plane": self.plane, "method": self.method, "assumption": self.assumption,
                "confidence": self.confidence, "needs": self.needs,
                "spread": round(self.spread, 3) if self.spread is not None and np.isfinite(self.spread) else
                (None if self.spread is None else 9.99)}


@dataclass
class Solution:
    walls: dict = field(default_factory=dict)   # wall key -> {"distance_in", "source", "facade", ...}
    lines: dict = field(default_factory=dict)   # line id -> Placed
    needs: str | None = None
    warning: str | None = None                  # the gravest check, in words (the editor's banner)
    pitch: dict | None = None                   # {"rise", "source"}: the pitch every face takes
    checks: list = field(default_factory=list)  # services/checks.py: what failed, in words
    reference: dict | None = None               # {"wall", "facade", "how"}: where the measurement went
    choice: dict | None = None                  # the wall choices tried, when any line was undecided
    penalty: float = 0.0
    camera: Camera | None = None                # the camera the solve used (the gables' rakes may pick another)
    camera_check: dict | None = None            # the cameras tried and what their rakes said
    reference_check: dict | None = None         # a second known length against the photo's

    def to_dict(self) -> dict:
        return {"walls": self.walls, "lines": {k: v.to_dict() for k, v in self.lines.items()}, "needs": self.needs,
                "warning": self.warning, "pitch": self.pitch, "checks": self.checks, "reference": self.reference,
                "choice": self.choice, "camera_check": self.camera_check, "reference_check": self.reference_check}


class Trace:
    """With SCALE_DEBUG, what each step of a solve saw and decided, and each line's lifecycle
    through it (services/scale_debug.py writes it out). Disabled, every call is a no-op."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.steps: dict[str, dict] = {}
        self.lines: dict[str, dict] = {}

    def step(self, name: str, **values) -> None:
        if self.enabled:
            self.steps.setdefault(name, {}).update(values)

    def add(self, name: str, key: str, value) -> None:
        if self.enabled:
            self.steps.setdefault(name, {}).setdefault(key, []).append(value)

    def line(self, line_id: str, **values) -> None:
        if self.enabled:
            self.lines.setdefault(line_id, {"id": line_id}).update(values)


# -- small geometry ---------------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else v


def _length(points3d: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points3d[:-1], points3d[1:])))


def _horizontal(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    return v - float(v @ up) * up


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


def _grazing(camera: Camera, pixels, normal: np.ndarray) -> float:
    """The sine of the shallowest angle at which the line's view rays meet the plane."""
    return min(abs(float(normal @ camera.ray(p))) for p in pixels)


def _roof_face(level: np.ndarray, through: np.ndarray, up: np.ndarray, angle: float,
               back: bool = False) -> tuple[np.ndarray, float]:
    """The roof face containing the level direction through a point, rising at `angle` into the
    house (away from the camera): (normal, offset). `back`: the face on the far side of the ridge,
    rising toward the camera instead."""
    level = _unit(_horizontal(level, up))
    inward = _unit(np.cross(up, level))
    if float(inward @ through) < 0:
        inward = -inward    # the roof rises away from the camera, into the house
    if back:
        inward = -inward
    normal = _unit(np.cross(level, np.cos(angle) * inward + np.sin(angle) * up))
    return normal, float(normal @ through)


def _overhang(line: Line | None) -> float:
    """How far in front of its wall a line's own plane is: an eave's overhang, else on the wall."""
    return EAVE_OVERHANG_IN if line is not None and line.edge == "eave" else 0.0


def _nearer(distance: float, by: float) -> float:
    """The offset of the parallel plane `by` inches nearer the camera than n·X = distance."""
    return distance - float(np.sign(distance)) * by


def _to_polyline(point, polyline) -> float:
    p = np.asarray(point, float)
    best = float("inf")
    for a, b in zip(polyline[:-1], polyline[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = np.clip(float((p - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-9), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def _on_eave(points, lines: list[Line], reach: float) -> bool:
    """Both tapped points on one eave: the salesperson measured the eave, not the wall below it."""
    return any(line.edge == "eave" and all(_to_polyline(p, line.points) <= reach for p in points)
               for line in lines)


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


# -- line relationships ------------------------------------------------------------------------------


def _ends_meet(a: list[float], b: list[float], reach: float) -> bool:
    return float(np.hypot(a[0] - b[0], a[1] - b[1])) <= reach


def _top_and_foot(line: Line) -> tuple[list[float], list[float]]:
    a, b = line.points[0], line.points[-1]
    return (a, b) if a[1] <= b[1] else (b, a)


def _span_px(line: Line) -> float:
    """End to end in the photo, px."""
    a, b = line.points[0], line.points[-1]
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def _long_enough(line: Line, reach: float) -> bool:
    """Long enough for its direction to count (MIN_LINE_REACHES)."""
    return _span_px(line) >= MIN_LINE_REACHES * reach


def _partner_rakes(rake: Line, lines: list[Line], reach: float) -> list[Line]:
    """The other rakes meeting this one at the top of both: a gable's other side. Looser on size
    than _rake_pairs (detection often stops one side early); the plane test confirms the pair."""
    top, _ = _top_and_foot(rake)
    span = _span_px(rake)
    out = []
    for other in lines:
        if other is rake or other.edge != "rake" or not _long_enough(other, reach):
            continue
        their_span = _span_px(other)
        if not span or not their_span or max(span, their_span) / min(span, their_span) > PARTNER_RATIO:
            continue
        if _ends_meet(top, _top_and_foot(other)[0], reach):
            out.append(other)
    return out


def _plane_pitch(camera: Camera, line: Line, normal: np.ndarray, up: np.ndarray) -> float:
    """The pitch (rise per 12) a sloped line implies lying on the vertical plane with this normal,
    at any distance: the plane's orientation alone decides it."""
    sign = 1.0 if float(normal @ camera.ray(line.points[0])) > 0 else -1.0
    points = [_on_plane(camera, p, normal, sign) for p in line.points]
    if any(p is None for p in points):
        return float("inf")
    v = points[-1] - points[0]
    run = float(np.linalg.norm(_horizontal(v, up)))
    return 12.0 * abs(float(v @ up)) / run if run > 1e-9 else float("inf")


def _plausible(pitch: float) -> bool:
    return bool(np.isfinite(pitch)) and PLAUSIBLE_PITCH[0] <= pitch <= PLAUSIBLE_PITCH[1]


def _rake_pairs(rake: Line, lines: list[Line], reach: float) -> list[tuple[list[float], list[float]]]:
    """The gable peaks a rake makes with other rakes: (its foot, the other rake's foot) for each."""
    ends = [rake.points[0], rake.points[-1]]
    span = float(np.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]))
    pairs = []
    if not _long_enough(rake, reach):
        return pairs
    for other in lines:
        if other is rake or other.edge != "rake" or not _long_enough(other, reach):
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


def _gable_keys(lines: list[Line], gables: dict[str, str], reach: float) -> dict[str, str]:
    """A wall key per gable end: rakes meeting at a peak share one, every other rake has its own.
    Two gable ends of a house face the same way but stand a house's width apart, so a facade's one
    distance cannot hold both ("side#1", "side#2")."""
    parent = {rake_id: rake_id for rake_id in gables}

    def root(rake_id: str) -> str:
        while parent[rake_id] != rake_id:
            rake_id = parent[rake_id]
        return rake_id

    by_id = {line.id: line for line in lines}
    for rake_id in gables:
        rake = by_id[rake_id]
        top, _ = _top_and_foot(rake)
        for other_id in gables:
            if other_id == rake_id or gables[other_id] != gables[rake_id]:
                continue
            other_top, _ = _top_and_foot(by_id[other_id])
            if _ends_meet(top, other_top, reach):
                parent[root(rake_id)] = root(other_id)
    keys: dict[str, str] = {}
    counts: dict[str, int] = {}
    named: dict[str, str] = {}
    for rake_id in gables:
        group = root(rake_id)
        if group not in named:
            counts[gables[rake_id]] = counts.get(gables[rake_id], 0) + 1
            named[group] = f"{gables[rake_id]}#{counts[gables[rake_id]]}"
        keys[rake_id] = named[group]
    return keys


def _reads_as_hip(rake: Line, lines: list[Line], reach: float) -> bool:
    """A rake rising from an eave's end to a ridge's or hip's end, with no other rake ending at its
    top, is a hip roof's corner. A gable's peak always has the other slope coming down from it,
    even when detection broke that slope into pieces of any size; a hip roof's corner never does."""
    top, foot = _top_and_foot(rake)

    def meets(end, edges) -> bool:
        return any(other is not rake and other.edge in edges and
                   any(_ends_meet(end, tip, reach) for tip in (other.points[0], other.points[-1]))
                   for other in lines)

    return not meets(top, ("rake",)) and meets(foot, ("eave", "junction")) and meets(top, ("ridge", "hip"))


def _hip_readings(lines: list[Line], facade_of: dict, reach: float, reference_id: str | None) -> dict[str, str]:
    """Rakes that are hips, with why: roof on both sides, two walls' eaves meeting at the foot, or
    the shape of a hip roof's corner.

    A gable's rake also ends at a corner where two walls' lines meet (its gable's base, often
    labelled an eave, and the next wall's eave), so a corner alone is not enough: roof on one side
    only ("free") says rake, and so does a top that meets another rake that is no hip (the
    gable's two rakes meeting at its peak)."""
    out, candidates = {}, {}
    for rake in lines:
        if rake.edge != "rake" or rake.id == reference_id:
            continue
        top, foot = _top_and_foot(rake)
        # A gable's rake has the other slope coming down from its top. A dormer's rake has roof on
        # both sides (its own and the slope behind it), and two gables meeting at an inside corner
        # put both rakes' feet where two walls' eaves meet (824fd558, 6c910b84): with a peak, still
        # gable rakes, not hips.
        at_peak = any(other is not rake and other.edge == "rake"
                      and any(_ends_meet(top, tip, reach) for tip in (other.points[0], other.points[-1]))
                      for other in lines)
        if rake.topology == "shared" and not at_peak:
            out[rake.id] = "shared"
            continue
        at_foot = {facade_of.get(e.id) for e in lines if e.edge in WALL_EDGES and facade_of.get(e.id)
                   and any(_ends_meet(foot, tip, reach) for tip in (e.points[0], e.points[-1]))}
        if len(at_foot) >= 2 and rake.topology != "free" and not at_peak:
            candidates[rake.id] = "corner"
        elif _reads_as_hip(rake, lines, reach):
            candidates[rake.id] = "shape"
    for rake in lines:
        why = candidates.get(rake.id)
        if why is None:
            continue
        top, _ = _top_and_foot(rake)
        partner = any(other is not rake and other.edge == "rake" and other.id not in candidates
                      and other.id not in out and _ends_meet(top, _top_and_foot(other)[0], reach)
                      for other in lines)
        if not partner:
            out[rake.id] = why
    return out


def _hip_bisector(alongs: list[np.ndarray], base: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, str]:
    """The hip's direction in plan from its corner: the two eaves' directions away from the
    corner, summed; with one, its direction plus the horizontal inward (away from the camera)."""
    if len(alongs) >= 2:
        return _unit(alongs[0] + alongs[1]), "the two eaves meeting there"
    inward = _unit(np.cross(up, alongs[0]))
    if float(inward @ base) < 0:
        inward = -inward
    return _unit(alongs[0] + inward), "its eave and the inward direction"


def _hip_holds(camera: Camera, rake: Line, lines: list[Line], facade_of: dict, normals: dict, up: np.ndarray,
               reach: float) -> bool:
    """Scale-free: placed as a hip over its foot's corner bisector, does the line imply a
    plausible pitch? (Every length scales with the wall's distance; the pitch does not.)"""
    top, foot = _top_and_foot(rake)
    base, alongs = None, []
    for other in lines:
        facade = facade_of.get(other.id)
        normal = normals.get(facade) if facade else None
        if other.edge not in WALL_EDGES or normal is None:
            continue
        for tip, far in ((other.points[0], other.points[-1]), (other.points[-1], other.points[0])):
            if not _ends_meet(foot, tip, reach):
                continue
            sign = 1.0 if float(normal @ camera.ray(tip)) > 0 else -1.0
            here, there = _on_plane(camera, tip, normal, sign), _on_plane(camera, far, normal, sign)
            if here is None or there is None:
                continue
            base = here if base is None else base
            along = _horizontal(there - here, up)
            if np.linalg.norm(along) > 1e-9 and all(abs(float(_unit(along) @ a)) < 0.9 for a in alongs):
                alongs.append(_unit(along))
    if base is None or not alongs:
        return False
    bisector, _ = _hip_bisector(alongs, base, up)
    normal = _unit(np.cross(bisector, up))
    point = _on_plane(camera, top, normal, float(normal @ base))
    if point is None:
        return False
    run = float(np.linalg.norm(_horizontal(point - base, up))) / np.sqrt(2)
    implied = 12.0 * float((point - base) @ up) / run if run > 1e-9 else float("inf")
    return PLAUSIBLE_PITCH[0] <= implied <= PLAUSIBLE_PITCH[1]


HIP_WHY = {
    "shared": "marked rake, but the roof shows on both its sides, so it is read as a hip. ",
    "corner": "marked rake, but two walls' eaves meet at its foot, so it is read as a hip. ",
    "shape": "marked rake, but no second rake meets it at a peak, so it is read as a hip; "
             "draw the other rake if this is a gable. ",
}


def _gable_facade(camera: Camera, rake: Line, lines: list[Line], reach: float, tolerance: float,
                  facades: list[str], facade_of: dict, normals: dict | None = None,
                  up: np.ndarray | None = None, follow_peak: bool = True) -> tuple[str | None, str | None]:
    """(facade, how) of the wall a rake lies on: from the rake itself (_gable_facade_own), else,
    when another rake comes down from its peak, that rake's gable wall ("peak"), since both slopes
    of a gable stand in one wall; an asymmetric gable's short side fails the pairing's size test
    (6c910b84: 477 px against 1233 px). None when nothing decides."""
    found = _gable_facade_own(camera, rake, lines, reach, tolerance, facades, facade_of, normals, up)
    if found[0] is not None or not follow_peak:
        return found
    top, _ = _top_and_foot(rake)
    for other in lines:
        if other is rake or other.edge != "rake":
            continue
        other_top, _ = _top_and_foot(other)
        if _ends_meet(top, other_top, reach):
            facade, _ = _gable_facade_own(camera, other, lines, reach, tolerance, facades, facade_of, normals, up)
            if facade is not None:
                return facade, "peak"
    return None, None


def _gable_facade_own(camera: Camera, rake: Line, lines: list[Line], reach: float, tolerance: float,
                      facades: list[str], facade_of: dict, normals: dict | None = None,
                      up: np.ndarray | None = None) -> tuple[str | None, str | None]:
    """_gable_facade from the rake itself.

    - "pair": with the other side of its gable, the plane on which both rakes imply one
      believable pitch (scale-free: only the plane's orientation matters). Where several planes
      pass, or none can be tested, the facade the two feet's joining line runs toward, then the
      side detection gave.
    - "base": an eave at its foot running under the gable (between its feet) is the gable's own
      base: the same facade.
    - "elimination": the eave at its foot runs away from the gable, so it is the next wall's: the
      gable is the other facade.
    - "plane": one rake alone, and only one facade's plane gives it a believable pitch.
    None when nothing decides (one facade and no pair: a side gable seen front-on)."""
    feet_vote = None
    for foot, other_foot in _rake_pairs(rake, lines, reach):
        if not camera.horizontal:
            return facades[0], "pair"        # no horizontal vanishing point: every wall faces the camera
        angles = [(angle, f) for angle, f in camera.facade_angles(foot, other_foot) if f in facades]
        if angles and angles[0][0] <= tolerance:
            feet_vote = angles[0][1]
            break

    testable = normals is not None and up is not None and bool(camera.horizontal)
    plausible_on: list[str] = []
    implausible_on: list[str] = []   # seen well enough to test, and no believable pitch there
    if testable:
        partners = _partner_rakes(rake, lines, reach)
        agreeing: list[tuple[float, str]] = []
        for facade in facades:
            normal = normals.get(facade)
            if normal is None or _grazing(camera, rake.points, normal) < GRAZING_MIN:
                continue
            mine = _plane_pitch(camera, rake, normal, up)
            if not _plausible(mine):
                implausible_on.append(facade)
                continue
            plausible_on.append(facade)
            theirs = [_plane_pitch(camera, p, normal, up) for p in partners]
            gaps = [abs(float(np.log(mine / t))) for t in theirs if _plausible(t)]
            if gaps and min(gaps) <= np.log(RAKE_PITCH_RATIO):
                agreeing.append((min(gaps), facade))
        if partners and agreeing:
            names = [f for _, f in agreeing]
            if len(names) == 1:
                return names[0], "pair"
            if feet_vote in names:
                return feet_vote, "pair"
            if rake.side in names:
                return rake.side, "pair"
            return min(agreeing)[1], "pair"
    if feet_vote is not None:
        return feet_vote, "pair"
    if len(facades) <= 1:
        return None, None

    # An eave at the foot: the gable's own base when it runs under the gable (toward the side its
    # top leans to), else the next wall's eave, the corner where the gable wall turns.
    top, foot = _top_and_foot(rake)
    lean = float(top[0]) - float(foot[0])
    base, turn = set(), set()
    for e in lines:
        facade = facade_of.get(e.id)
        if e.edge not in WALL_EDGES or not facade:
            continue
        for tip, far in ((e.points[0], e.points[-1]), (e.points[-1], e.points[0])):
            if _ends_meet(foot, tip, reach):
                (base if (float(far[0]) - float(foot[0])) * lean > 0 else turn).add(facade)
            elif _ends_meet(top, tip, reach):
                turn.add(facade)
    found, how = None, None
    if len(base) == 1:
        found, how = next(iter(base)), "base"
    elif len(turn) == 1:
        others = [f for f in facades if f not in turn]
        if len(others) == 1:
            found, how = others[0], "elimination"
    if testable and (found is None or found in implausible_on) and plausible_on:
        # The plane test overrules a reading whose plane, seen well, gives this rake no believable
        # pitch (a gable wall seen edge-on cannot be tested, and stands).
        if rake.side in plausible_on:
            return rake.side, "plane"
        if len(plausible_on) == 1:
            return plausible_on[0], "plane"
        return None, None
    if found is None and rake.side in facades and (not testable or rake.side in plausible_on):
        return rake.side, "given"
    return found, how


def _feet(inches: float) -> str:
    return f"{inches / 12:.0f} ft" if inches >= 24 else f"{inches:.0f} in"


def _wall_reading(camera: Camera, walls: dict, normals: dict, placed: dict, by_id: dict, line: Line) -> float | None:
    """What a line measures if it lies on the nearest wall (the wall of the placed line closest to
    it across the photo): only a rough reading for a sloped line, enough to catch a slip."""
    xs = [p[0] for p in line.points]
    best = None
    for other_id, done in placed.items():
        if other_id == line.id or done.wall not in walls or other_id not in by_id:
            continue
        theirs = [p[0] for p in by_id[other_id].points]
        gap = max(0.0, min(theirs) - max(xs), min(xs) - max(theirs))
        if best is None or gap < best[0]:
            best = (gap, done.wall)
    if best is None:
        return None
    points = [camera.on_wall(p, normals[best[1]], walls[best[1]]["distance_in"]) for p in line.points]
    return None if any(p is None for p in points) else _length(points)


def _questioned(typed: float | None, suggested: float | None) -> str | None:
    """A sentence when a typed length is far from what the photo suggests; None otherwise."""
    if not typed or not suggested or 1 / TYPED_LENGTH_RATIO <= typed / suggested <= TYPED_LENGTH_RATIO:
        return None
    return (f"{_feet(typed)} seems {'short' if typed < suggested else 'long'} for this line; "
            f"the photo suggests about {_feet(suggested)}")


# -- preparation: everything that does not depend on the wall choice ----------------------------------


@dataclass
class _Prep:
    camera: Camera
    lines: list[Line]
    by_id: dict
    reach: float
    tolerance: float
    facades: list[str]
    normals: dict
    up: np.ndarray
    assign: dict
    reference: dict | None
    ref_assign: walls_module.Assignment | None
    rise: float
    rise_source: str
    focal_note: str | None
    overrides: dict
    features: list
    rake_check: dict = field(default_factory=dict)


def _prepare(camera: Camera, lines: list[Line], measurement: dict | None, interpreted: dict | None,
             tolerance: float, overrides: dict, features: list, trace: Trace) -> _Prep:
    reach = CORNER_REACH_FRACTION * max(camera.width, camera.height)
    by_id = {line.id: line for line in lines}
    facades = list(camera.facades) or ["front"]
    dropped = None
    if camera.focal_checks.get("second_facade") == "found by its focal sweep" and len(facades) > 1:
        # A facade the camera found by sweeping segments must be this house's: some level line of
        # the house runs toward it, or a gable's two rake feet do (a gable end has no eave).
        # Otherwise a neighbour's roof, a fence or a driveway built it.
        swept = facades[-1]
        level = any(line.edge in LEVEL and (line.side == swept or camera.facade_of(line.points[0], line.points[-1], tolerance) == swept)
                    for line in lines)
        gable = any(camera.facade_of(foot, other, tolerance) == swept
                    for line in lines if line.edge == "rake" for foot, other in _rake_pairs(line, lines, reach))
        if not level and not gable:
            facades.remove(swept)
            dropped = swept
    normals = {f: camera.wall_normal(f) for f in facades}
    up = world_up(camera)
    trace.step("1_camera", **camera.to_dict(), up=up.tolist(), reach_px=reach, facades_used=facades,
               facade_dropped=dropped, wall_normals={f: (n.tolist() if n is not None else None) for f, n in normals.items()})
    for line in lines:
        trace.line(line.id, edge=line.edge, side_given=line.side, topology=line.topology,
                   points=[[round(float(x), 1), round(float(y), 1)] for x, y in line.points])

    assign = walls_module.assign_lines(camera, lines, facades, tolerance, reach)
    for line_id, found in assign.items():
        trace.line(line_id, side_assignment=found.to_dict())
    trace.step("2_sides", margin_min_deg=walls_module.MARGIN_MIN_DEG, tolerance_deg=tolerance,
               lines={line_id: found.to_dict() for line_id, found in assign.items()})

    reference = ref_assign = None
    if measurement:
        given = measurement.get("facade") if measurement.get("facade_given", True) else None
        region_id = measurement.get("region_id")
        if region_id:
            line = by_id.get(region_id)
            pixels = line.points if line else None
            on_eave = line is not None and line.edge == "eave"
            if line is not None and line.id in assign and not given:
                ref_assign = assign[line.id]
            else:
                facade = given or measurement.get("facade")
                ref_assign = walls_module.Assignment(facade, "given" if given else "line", None, [facade] if facade else [])
        else:
            pixels = measurement.get("points")
            on_eave = bool(pixels) and _on_eave(pixels, lines, reach)
            ref_assign = (walls_module.assign_points(camera, pixels, lines, assign, facades, tolerance, reach, given)
                          if pixels else None)
        reference = {"pixels": pixels, "length_in": float(measurement["length_in"]), "region_id": region_id,
                     "on_eave": on_eave, "given": given, "hint": measurement.get("facade")}
        trace.step("3_measurement", by="line" if region_id else "points", region_id=region_id, pixels=pixels,
                   length_in=measurement.get("length_in"), facade_given=given,
                   side_assignment=ref_assign.to_dict() if ref_assign else None, on_eave=on_eave)

    for feature in features:
        found = walls_module.assign_points(camera, [feature["left"], feature["right"]], lines, assign, facades,
                                           tolerance, reach)
        feature["facade"] = found.facade if found.how not in ("ambiguous", "none") else None
        feature["side_assignment"] = found.to_dict()

    # Pitch from the rakes, before any scale: a rake's slope on its gable wall does not depend on
    # how far that wall is.
    facade_of = {line_id: found.facade or (found.candidates[0] if found.candidates else None)
                 for line_id, found in assign.items()}
    as_hip = _hip_readings(lines, facade_of, reach, (measurement or {}).get("region_id"))
    peak, lone = [], []
    readings: list[tuple[Line, str, float]] = []   # every rake read on a gable wall, believable or not
    for rake in lines:
        if rake.edge != "rake":
            continue
        # Read as a hip by its shape alone, it is still a rake (and measures the pitch) when as a
        # hip it would imply no believable pitch: a side gable's lone rake, seen from the front.
        if rake.id in as_hip and (as_hip[rake.id] != "shape" or _hip_holds(camera, rake, lines, facade_of, normals, up, reach)):
            trace.add("5_pitch", "rakes", {"id": rake.id, "used": False, "why": f"read as a hip ({as_hip[rake.id]})"})
            continue
        if not _long_enough(rake, reach):
            trace.add("5_pitch", "rakes", {"id": rake.id, "used": False, "why": f"too short to vote ({_span_px(rake):.0f} px)"})
            continue
        # The camera check reads each rake on the wall its structure says, before any plane test:
        # the plane test picks a plane where the pitch is believable, which would hide a camera
        # whose walls are turned wrong.
        structural, _ = _gable_facade(camera, rake, lines, reach, tolerance, facades, facade_of)
        if structural is not None and normals.get(structural) is not None \
                and _grazing(camera, rake.points, normals[structural]) >= GRAZING_MIN:
            readings.append((rake, structural, _plane_pitch(camera, rake, normals[structural], up)))
        gable, how = _gable_facade(camera, rake, lines, reach, tolerance, facades, facade_of, normals, up)
        normal = normals.get(gable) if gable else None
        if normal is None:
            trace.add("5_pitch", "rakes", {"id": rake.id, "used": False, "why": "no gable wall"})
            continue
        grazing = _grazing(camera, rake.points, normal)
        if grazing < GRAZING_MIN:
            trace.add("5_pitch", "rakes", {"id": rake.id, "used": False, "gable": gable,
                                           "why": f"gable wall seen edge-on ({np.degrees(np.arcsin(min(1.0, grazing))):.1f} deg)"})
            continue
        pitch = _plane_pitch(camera, rake, normal, up)
        if not PLAUSIBLE_PITCH[0] <= pitch <= PLAUSIBLE_PITCH[1]:
            trace.add("5_pitch", "rakes", {"id": rake.id, "used": False, "gable": gable,
                                           "rise": round(pitch, 2) if np.isfinite(pitch) else None, "why": "implausible pitch"})
            continue
        at_peak = bool(_rake_pairs(rake, lines, reach))
        trace.add("5_pitch", "rakes", {"id": rake.id, "used": True, "gable": gable, "gable_how": how,
                                       "rise": round(pitch, 2), "at_peak": at_peak})
        (peak if at_peak else lone).append(pitch)
    chosen = peak or lone
    if chosen:
        rise, rise_source = float(np.median(chosen)), "measured"
    else:
        rise, rise_source = pitch_hint(interpreted)
    rake_check = _rake_check(readings, reach)
    # Lines this camera can put on a wall at all: eaves and junctions with a side, rakes with a
    # gable wall. A camera missing a wall the house shows (a corner shot read as front-on) leaves
    # that wall's lines with none.
    rake_check["walled"] = (sum(1 for line in lines if line.edge in WALL_EDGES and facade_of.get(line.id))
                            + len(readings))
    trace.step("5_pitch", used={"rise": round(rise, 1), "source": rise_source}, image_estimate=pitch_hint(interpreted),
               camera_check=rake_check)
    focal_note = None if camera.focal_source != "assumed" else "the lens is assumed, so depth is approximate"
    return _Prep(camera, lines, by_id, reach, tolerance, facades, normals, up, assign, reference, ref_assign,
                 rise, rise_source, focal_note, overrides, features, rake_check)


# Two rakes of one gable read on its wall agree on the pitch to within this ratio when the wall
# is where the camera says: a wall turned the wrong way squeezes one side's run more than the other.
RAKE_PITCH_RATIO = 1.5


def _rake_check(readings: list[tuple[Line, str, float]], reach: float) -> dict:
    """The gables' rakes as a test of the camera, before any scale: a rake's pitch on its gable
    wall depends only on the wall's orientation, which the focal length and the vanishing points
    set. `bad` counts rakes implying an implausible pitch and gable pairs that disagree; `good`
    the rakes implying a believable one."""
    good = sum(1 for _, _, pitch in readings if PLAUSIBLE_PITCH[0] <= pitch <= PLAUSIBLE_PITCH[1])
    implausible = len(readings) - good
    disagree = []
    for i, (a, gable_a, pitch_a) in enumerate(readings):
        for b, gable_b, pitch_b in readings[i + 1:]:
            if gable_a != gable_b or not (np.isfinite(pitch_a) and np.isfinite(pitch_b)) or min(pitch_a, pitch_b) <= 0:
                continue
            if not _ends_meet(_top_and_foot(a)[0], _top_and_foot(b)[0], reach):
                continue
            if max(pitch_a, pitch_b) / min(pitch_a, pitch_b) > RAKE_PITCH_RATIO:
                disagree.append([a.id, b.id])
    return {"good": good, "bad": implausible + len(disagree), "implausible": implausible, "disagree": disagree}


# -- the solve ---------------------------------------------------------------------------------------


def solve(camera: Camera, lines: list[Line], measurement: dict | None, interpreted: dict | None = None,
          tolerance_deg: float = 3.0, overrides: dict | None = None, trace: Trace | None = None,
          features: list | None = None, check: dict | None = None) -> Solution:
    """Scale and 3-D for every line. `measurement` is Photo.scale's {facade, facade_given, points
    or region_id, length_in}; `overrides` is {line id: {"mid": [x, y], "length_in"}} for lines the
    geometry cannot place; `features` are detected doors and garage doors ({"id", "kind", "label",
    "top", "bottom", "left", "right"} in pixels), checked for standard sizes. `trace` records every
    step (SCALE_DEBUG). The solve may use another camera than `camera` when the gables' rakes
    reject it (Solution.camera, Solution.camera_check)."""
    trace = trace if trace is not None else Trace(enabled=False)
    camera, camera_check = _checked_camera(camera, lines, measurement, interpreted, tolerance_deg,
                                           overrides or {}, features or [])
    prep = _prepare(camera, lines, measurement, interpreted, tolerance_deg, overrides or {},
                    [dict(f) for f in (features or [])], trace)
    trace.step("1_camera", camera_check=camera_check)
    solution = _choose(prep, trace)
    solution.camera, solution.camera_check = camera, camera_check
    if check and check.get("region_id"):
        _second_reference(solution, check, trace)
    return solution


# A second known length farther than this from the photo's means one of the two is off.
SECOND_REFERENCE_SLACK = 0.05


def _second_reference(solution: Solution, check: dict, trace: Trace) -> None:
    """The second known length against what the solve made of its line: the one test of the scale
    that needs no assumption about houses."""
    typed = float(check["length_in"])
    done = solution.lines.get(check["region_id"])
    solved = done.length_in if done is not None and done.points3d and done.method != "override" else None
    result = {"region_id": check["region_id"], "typed_in": round(typed, 1),
              "solved_in": round(solved, 1) if solved else None, "off": None}
    if solved:
        off = solved / typed - 1.0
        result["off"] = round(off, 3)
        if abs(off) > SECOND_REFERENCE_SLACK:
            message = (f"Your second length ({_feet(typed)}) and the scale disagree: the photo makes that line "
                       f"{_feet(solved)}, {abs(off) * 100:.0f}% {'longer' if off > 0 else 'shorter'}. One of the two "
                       "lengths, or the side it is on, is off.")
            solution.checks.append({"kind": "second_reference", "lines": [check["region_id"]], "message": message})
            solution.warning = solution.warning or message
    trace.step("9_result", reference_check=result)
    solution.reference_check = result


# A second facade read off the roof's own level lines (an eave that runs toward no vanishing point
# the camera has) is taken only when a second reading backs it: another such line, a vanishing-
# point pair the conditioning test set aside, or the lens's own EXIF, within this.
LINE_FOCAL_AGREE = 0.10
LINE_LEVEL_MAX_DEG = 45.0          # steeper than this in the photo, a line is not read as level


def _focal_from_lines(camera: Camera, lines: list[Line], tolerance: float) -> tuple[float, dict] | None:
    """(focal, 35 mm, and the evidence) when the photo shows one facade and some of its eaves run
    toward none: each such eave meets the horizon at the other facade's vanishing point, and that
    point with the first gives the focal length (f² = −(v1 − c)·(v2 − c)).

    Measured on a corner photo (824fd558): a level camera, one facade found, the focal assumed at
    26 mm; the left gable's eave put the other vanishing point at a 45.0 mm lens, and the
    vertical pair the conditioning test had set aside said 47.3 mm. Under the assumed 26 mm the
    square facade ran 82 degrees off that eave, so the gable had no wall at all."""
    horizontal, facades = camera_module.observed(camera)
    if len(horizontal) != 1 or not horizontal[0][2]:
        return None
    centre = np.asarray(camera.centre, float)
    v1 = np.array(horizontal[0][:2], float)
    # The horizon passes through the facade's vanishing point, square to the line from the
    # principal point to the vertical's (exact for a pinhole with that principal point).
    if camera.vertical is not None:
        vertical = np.asarray(camera.vertical, float)
        toward = vertical[:2] - centre * vertical[2] if vertical[2] else vertical[:2]
        updir = _unit(toward)
    else:
        updir = np.array([0.0, -1.0])
    across = np.array([-updir[1], updir[0]])
    long_edge = float(max(camera.width, camera.height))
    readings = []
    for line in lines:
        if line.edge not in WALL_EDGES or len(line.points) < 2:
            continue
        a, b = np.asarray(line.points[0], float), np.asarray(line.points[-1], float)
        d = b - a
        if np.linalg.norm(d) < 1.0 or abs(float(_unit(d) @ updir)) > np.sin(np.radians(LINE_LEVEL_MAX_DEG)):
            continue
        middle = (a + b) / 2
        toward_v1 = _unit(v1 - middle)
        if np.degrees(np.arccos(min(1.0, abs(float(_unit(d) @ toward_v1))))) <= tolerance:
            continue                    # it runs toward the facade the camera has
        matrix = np.column_stack([d, -across])
        if abs(np.linalg.det(matrix)) < 1e-9:
            continue
        t, _ = np.linalg.solve(matrix, v1 - a)
        v2 = [*map(float, a + t * d), 1.0]
        focal = camera_module._focal_from_pair(list(horizontal[0]), v2, centre)
        if focal is None:
            continue
        f35 = camera_module._to_35mm(focal, long_edge)
        if not camera_module.PLAUSIBLE_FOCAL_35MM[0] <= f35 <= camera_module.PLAUSIBLE_FOCAL_35MM[1]:
            continue
        if camera_module.focal_spread(list(horizontal[0]), v2, centre) > camera_module.FOCAL_SPREAD_MAX:
            continue
        readings.append((f35, line.id))
    if not readings:
        return None
    others = [("set-aside vanishing-point pair", float(r["focal_35mm"]))
              for r in camera.focal_checks.get("vp_rejected", []) if r.get("focal_35mm")]
    if camera.focal_checks.get("exif"):
        others.append(("EXIF", float(camera.focal_checks["exif"])))
    best = None
    for f35, _ in readings:
        agree = [r for r in readings if abs(r[0] - f35) <= LINE_FOCAL_AGREE * f35]
        backing = [(name, value) for name, value in others if abs(value - f35) <= LINE_FOCAL_AGREE * f35]
        if len(agree) < 2 and not backing:
            continue
        if best is None or (len(agree), len(backing)) > (len(best[0]), len(best[1])):
            best = (agree, backing)
    if best is None:
        return None
    agree, backing = best
    focal_35mm = float(np.median([r[0] for r in agree]))
    return focal_35mm, {"focal_35mm": round(focal_35mm, 1), "lines": [r[1] for r in agree],
                        "readings_35mm": [round(r[0], 1) for r in agree],
                        "backed_by": [{"by": name, "focal_35mm": round(value, 1)} for name, value in backing]}


def alternatives(camera: Camera, lines: list[Line] | None = None, tolerance: float = 3.0) -> list[tuple[str, Camera]]:
    """Other cameras the same vanishing points allow, for the gables' rakes to judge:
    - a focal read off the vanishing points can be noise the conditioning test let through: the
      lens's own (EXIF), else a typical phone's, with the second facade made square under it;
    - with one facade seen and an assumed focal, no second facade was made: one made square
      under the assumed focal, kept only when its gables' rakes confirm it;
    - with one facade seen, eaves that run toward none give the other facade and the focal
      length themselves (_focal_from_lines), when a second reading backs them."""
    out = []
    if camera.focal_source == "vp":
        exif = camera.focal_checks.get("exif")
        focal, source = (float(exif), "exif") if exif else (camera_module.ASSUMED_FOCAL_35MM, "assumed")
        if abs(focal - camera.focal_35mm) > 0.15 * camera.focal_35mm:
            out.append((f"{source} focal {focal:.0f} mm", camera_module.with_focal(camera, focal, source)))
    elif len(camera.horizontal) == 1:
        out.append(("second facade square at the assumed focal",
                    camera_module.with_focal(camera, camera.focal_35mm, camera.focal_source)))
    found = _focal_from_lines(camera, lines or [], tolerance)
    if found is not None and abs(found[0] - camera.focal_35mm) > LINE_FOCAL_AGREE * camera.focal_35mm:
        other = camera_module.with_focal(camera, found[0], "vp")
        other.focal_checks["second_facade_from_lines"] = found[1]
        out.append((f"second facade from the roof lines ({found[0]:.0f} mm)", other))
    return out


def _checked_camera(camera: Camera, lines: list[Line], measurement: dict | None, interpreted: dict | None,
                    tolerance: float, overrides: dict, features: list) -> tuple[Camera, dict]:
    """The camera the gables' rakes believe. Each rake read on its gable wall implies a pitch with
    no scale at all, so a camera whose walls are turned the wrong way shows at once: pitches
    beyond any house's, or a gable's two rakes disagreeing. Another camera is used only when
    its rakes say fewer such things and no fewer believable ones, or more believable ones and
    no more such things, or as many believable ones and more lines on a wall."""
    def judged(cam: Camera) -> dict:
        prep = _prepare(cam, lines, measurement, interpreted, tolerance, overrides,
                        [dict(f) for f in features], Trace(enabled=False))
        return prep.rake_check

    own = judged(camera)
    check = {"photo": {"focal_35mm": round(camera.focal_35mm, 1), "source": camera.focal_source, **own},
             "tried": [], "used": "photo"}
    best, best_check = camera, own
    for name, other in alternatives(camera, lines, tolerance):
        verdict = judged(other)
        check["tried"].append({"name": name, "focal_35mm": round(other.focal_35mm, 1), **verdict})
        fewer_bad = verdict["bad"] < best_check["bad"] and verdict["good"] >= best_check["good"]
        more_good = verdict["good"] > best_check["good"] and verdict["bad"] <= best_check["bad"]
        # A wall the house shows and this camera has: more of its lines on a wall, its rakes no worse.
        more_walled = (verdict.get("walled", 0) > best_check.get("walled", 0)
                       and verdict["bad"] <= best_check["bad"] and verdict["good"] >= best_check["good"])
        if fewer_bad or more_good or more_walled:
            best, best_check, check["used"] = other, verdict, name
    return best, check


def _choose(prep: _Prep, trace: Trace) -> Solution:
    """The solve over every choice of the undecided lines' walls, keeping the most consistent."""
    # The undecided: the reference first, then the longest lines, as many as MAX_CHOICES allows.
    items: list[tuple[str, list[str]]] = []
    if prep.reference and prep.ref_assign and prep.ref_assign.how == "ambiguous" and not prep.reference["region_id"]:
        items.append((REFERENCE, prep.ref_assign.candidates))
    undecided = sorted((line_id for line_id, found in prep.assign.items()
                        if found.how == "ambiguous" and len(found.candidates) > 1),
                       key=lambda line_id: -_length([np.array([*p, 0.0]) for p in prep.by_id[line_id].points]))
    items += [(line_id, prep.assign[line_id].candidates) for line_id in undecided]
    while items and int(np.prod([len(c) for _, c in items])) > MAX_CHOICES:
        items.pop()
    if not items or not prep.reference:
        solution = _core(prep, {}, trace)
        trace.step("7_choice", items=[], tried=[])
        _uncertainty(prep, {}, solution, trace)
        _estimate_from_neighbours(prep.lines, solution, prep.reach)
        return solution

    tried = []
    for values in product(*(candidates for _, candidates in items)):
        choice = {key: value for (key, _), value in zip(items, values)}
        quiet = _core(prep, choice, Trace(enabled=False))
        # The preferred candidate (a given side, else the nearest vanishing point) breaks ties.
        preference = 0.01 * sum(1 for (key, candidates), value in zip(items, values) if value != candidates[0])
        tried.append((quiet.penalty + preference, choice, len(quiet.checks)))
    score, best, _ = min(tried, key=lambda t: t[0])
    solution = _core(prep, best, trace)
    solution.choice = {"items": [{"id": key, "candidates": candidates} for key, candidates in items],
                       "chosen": best, "tried": [{"choice": c, "score": round(s, 3), "checks": n} for s, c, n in tried]}
    trace.step("7_choice", **solution.choice)
    _uncertainty(prep, best, solution, trace)
    _estimate_from_neighbours(prep.lines, solution, prep.reach)
    return solution


def _with_camera(prep: _Prep, cam: Camera) -> _Prep:
    """The same preparation (sides, reference, pitch) under another camera: only the geometry moves."""
    return replace(prep, camera=cam, normals={f: cam.wall_normal(f) for f in prep.facades}, up=world_up(cam))


def _variants(prep: _Prep) -> list[tuple[str, _Prep]]:
    """The solve's inputs, each moved by the error the photo itself allows: every vanishing point
    turned by VP_JITTER_DEG either way, the focal length by its doubt, the pitch by a step."""
    cam = prep.camera
    out = []
    _, seen = camera_module.observed(cam)
    for which in [*seen, *(["vertical"] if cam.vertical is not None else [])]:
        for degrees in (-VP_JITTER_DEG, VP_JITTER_DEG):
            moved = camera_module.perturbed(cam, which, degrees=degrees)
            if moved is not None:
                out.append((f"{which} {degrees:+.1f} deg", _with_camera(prep, moved)))
    doubt = FOCAL_DOUBT.get(cam.focal_source, 0.1)
    for scale in (1.0 - doubt, 1.0 + doubt):
        moved = camera_module.perturbed(cam, "focal", focal_scale=scale)
        if moved is not None:
            out.append((f"focal x{scale:.2f}", _with_camera(prep, moved)))
    step = PITCH_DOUBT.get(prep.rise_source, 2.0)
    for change in (-step, step):
        if prep.rise + change > 0.5:
            out.append((f"pitch {change:+.0f}/12", replace(prep, rise=prep.rise + change)))
    return out


def _reference_error(prep: _Prep) -> float:
    """Relative: the reference tapped a pixel off at each end. Every length scales with it."""
    pixels = (prep.reference or {}).get("pixels")
    if not pixels or len(pixels) < 2:
        return 0.0
    length = sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(pixels[:-1], pixels[1:]))
    return 2.0 / length if length > 0 else 0.0


def _uncertainty(prep: _Prep, choice: dict, solution: Solution, trace: Trace) -> None:
    """Each placed length's spread: the solve run again under every variant (same sides, same wall
    choice), the largest relative move, combined with the reference's own tapping error. A length
    moving more than SPREAD_MAX is shown as approximate (assumed, with how far it moves): still the
    photo's best reading of it. A line that some variant cannot place at all is as unstable as can
    be, and is not shown ("unreliable"); _estimate_from_neighbours then reads it off the lines
    around it."""
    base = {line_id: done.length_in for line_id, done in solution.lines.items()
            if done.length_in and done.points3d and done.method not in ("override", "measured")}
    reference_error = _reference_error(prep)
    variants = _variants(prep)
    worst = {line_id: 0.0 for line_id in base}
    for _, variant in variants:
        other = _core(variant, choice, Trace(enabled=False))
        for line_id, length in base.items():
            done = other.lines.get(line_id)
            moved = (abs(done.length_in - length) / length if done is not None and done.length_in and done.points3d
                     else float("inf"))
            worst[line_id] = max(worst[line_id], moved)
    refused = []
    for line_id, geometric in worst.items():
        spread = float(np.hypot(geometric, reference_error)) if np.isfinite(geometric) else float("inf")
        done = solution.lines[line_id]
        done.spread = spread
        if spread > SPREAD_MAX:
            refused.append(line_id)
            trace.line(line_id, refused="its length moves too much under the photo's own errors",
                       refused_length_in=round(done.length_in, 1))
            if np.isfinite(spread):
                # Still the photo's best reading of it: shown as approximate, with how far it can
                # move, rather than withheld (the salesperson asked for an approximate length where
                # the measurement is unsure). A hand-set length replaces it.
                note = f"approximate: its length moves up to {spread:.0%} under the photo's own small errors"
                done.assumption = f"{note}; {done.assumption}" if done.assumption else note
                done.confidence = ASSUMED
            else:
                solution.lines[line_id] = Placed(line_id, needs="unreliable", wall=done.wall, side=done.side,
                                                 side_how=done.side_how, spread=spread)
        trace.line(line_id, spread=round(spread, 3) if np.isfinite(spread) else None)
    for line_id, done in solution.lines.items():
        if done.method == "measured":
            done.spread = 0.0
    trace.step("9_result", uncertainty={"reference_error": round(reference_error, 3),
                                        "variants": [name for name, _ in variants],
                                        "spread_max": SPREAD_MAX, "refused": refused})


def _core(prep: _Prep, choice: dict, trace: Trace) -> Solution:
    camera, lines, by_id, reach, up = prep.camera, prep.lines, prep.by_id, prep.reach, prep.up
    solution = Solution(pitch={"rise": round(prep.rise, 1), "source": prep.rise_source})
    angle = float(np.arctan2(prep.rise, 12.0))
    pitch_words = {"measured": f"the {prep.rise:.1f}/12 pitch measured on the gable rakes",
                   "estimated": f"an estimated {prep.rise:g}/12 pitch"}.get(prep.rise_source,
                                                                            f"an assumed {prep.rise:g}/12 pitch")
    pitch_sure = STATED if prep.rise_source == "measured" else ASSUMED

    # 1. Each level line's facade, the undecided ones as chosen.
    facade_of: dict[str, str | None] = {}
    side_how: dict[str, str] = {}
    for line_id, found in prep.assign.items():
        if found.how == "ambiguous":
            facade_of[line_id] = choice.get(line_id, found.candidates[0] if found.candidates else None)
            side_how[line_id] = "chosen" if line_id in choice else "ambiguous"
        else:
            facade_of[line_id], side_how[line_id] = found.facade, found.how

    # 2. Walls: each facade's main wall, and the set-back walls standing on roofs.
    setbacks = walls_module.setbacks(lines, facade_of, reach)
    key_of = {line_id: f for line_id, f in facade_of.items() if f and by_id[line_id].edge in WALL_EDGES}
    for setback in setbacks:
        for member in setback.members:
            key_of[member] = setback.key
    # A facade's separate runs of wall lines are separate walls ("front", "front.2"): a garage
    # wing's face and the entry's each get their own distance, from their own corners.
    key_of, split_of = walls_module.physical_walls(lines, key_of, prep.facades, reach)
    wall_facade = {f: f for f in prep.facades}
    wall_facade.update({s.key: s.facade for s in setbacks})
    wall_facade.update(split_of)
    normals = {key: prep.normals.get(f) for key, f in wall_facade.items()}
    trace.step("4_walls", setbacks=[s.to_dict() for s in setbacks],
               physical_walls={key: sorted(line_id for line_id, k in key_of.items() if k == key)
                               for key in [*prep.facades, *split_of]})

    ref = prep.reference
    reference_id = ref["region_id"] if ref else None

    # Each line's wall: its own for eaves and junctions, its gable's for rakes (before the
    # measurement: a measured rake is on its gable wall).
    as_hip = _hip_readings(lines, facade_of, reach, reference_id)
    wall_of: dict[str, str] = dict(key_of)
    gable_how: dict[str, str] = {}
    gables: dict[str, str] = {}
    for line in lines:
        if line.id in as_hip:
            trace.line(line.id, read_as_hip=as_hip[line.id])
        if line.edge == "rake" and line.id not in as_hip:
            gable, how = _gable_facade(camera, line, lines, reach, prep.tolerance, prep.facades, facade_of,
                                       prep.normals, up)
            trace.line(line.id, gable_found=gable, gable_how=how)
            if gable is not None and normals.get(gable) is not None:
                gables[line.id] = gable
                gable_how[line.id] = how
    gable_key = _gable_keys(lines, gables, reach)
    # A gable standing on a roof (a dormer's front): its feet over the junction where that wall
    # meets the roof, in the same direction, put it in that set-back wall (6c910b84: two dormers'
    # rakes had no wall at all while the wall they stand on was reached through the roof face).
    for group in set(gable_key.values()):
        members = [rake_id for rake_id, key in gable_key.items() if key == group]
        facade = gables[members[0]]
        feet = [_top_and_foot(by_id[rake_id])[1] for rake_id in members]
        x0, x1 = min(f[0] for f in feet), max(f[0] for f in feet)
        foot_y = max(f[1] for f in feet)
        best = None
        for setback in setbacks:
            if setback.facade != facade:
                continue
            # Most of the feet over the junction: detection often stops a junction short of a
            # dormer's far side (6c910b84: the second dormer, 63% over it).
            low, high = setback.span[0] - reach, setback.span[1] + reach
            if x1 - x0 < 1.0:
                if not low <= x0 <= high:     # one rake: its foot over the junction
                    continue
            elif min(x1, high) - max(x0, low) < 0.5 * (x1 - x0):
                continue
            gap = walls_module._y_at(by_id[setback.junction].points, (x0 + x1) / 2) - foot_y
            if gap > -reach and (best is None or gap < best[0]):
                best = (gap, setback.key)
        if best is not None:
            for rake_id in members:
                gable_key[rake_id] = best[1]
                gable_how[rake_id] = f"{gable_how.get(rake_id)}, standing on the roof"

    def register_gable(rake_id: str, facade: str, key: str) -> None:
        gable_key[rake_id] = key
        wall_of[rake_id] = key
        wall_facade[key] = facade
        normals[key] = prep.normals.get(facade)

    for rake_id, key in list(gable_key.items()):
        register_gable(rake_id, gables[rake_id], key)
    # A dormer's side walls: an eave running back from the foot of a dormer's rake is the eave of
    # that dormer's own side wall. Two dormers have four such walls facing one way at four depths,
    # so each gets its own plane, reached through that corner, never the facade's main wall
    # (6c910b84: three of them on one plane ended 29 to 32 ft off the dormers they meet).
    on_roof = {setback.key for setback in setbacks}
    dormer_feet = [_top_and_foot(by_id[rake_id])[1] for rake_id, key in gable_key.items() if key in on_roof]
    for line in lines:
        facade = facade_of.get(line.id)
        if line.edge != "eave" or not facade or wall_of.get(line.id) in on_roof:
            continue
        if any(_ends_meet(end, foot, reach) for end in (line.points[0], line.points[-1]) for foot in dormer_feet):
            key = f"{facade}@{line.id[:8]}"
            key_of[line.id] = wall_of[line.id] = key
            wall_facade[key] = facade
            normals[key] = prep.normals.get(facade)

    # 3. The measurement fixes its wall's distance.
    walls: dict = solution.walls
    measured = None
    reference_on_eave = bool(ref and ref["on_eave"])
    if ref is None:
        solution.needs = "measurement"
        trace.step("3_measurement", needs="measurement")
    else:
        if reference_id and reference_id in gable_key:
            # A measured rake lies in its gable wall, whatever side was picked or labelled: the
            # photo decides it (6c910b84: a gable-end rake picked as "right", the long side wall,
            # put the whole house's scale on the wrong plane).
            facade, how, key = gables[reference_id], "gable", gable_key[reference_id]
            if ref["given"] and ref["hint"] and ref["hint"] != facade:
                how = f"gable (not the {ref['hint']} side picked: a rake lies in its gable wall)"
        elif reference_id and reference_id in facade_of and facade_of[reference_id] and not ref["given"]:
            facade, how = facade_of[reference_id], side_how.get(reference_id)
            key = key_of.get(reference_id, facade)
        else:
            found = prep.ref_assign
            if found is not None and found.facade:
                facade, how = found.facade, found.how
            elif found is not None and found.how == "ambiguous" and found.candidates:
                facade = choice.get(REFERENCE, found.candidates[0])
                how = "chosen" if REFERENCE in choice else "ambiguous"
            else:
                facade, how = ref["hint"] or prep.facades[0], "hint"
            setback = (walls_module.setback_of_points(ref["pixels"], lines, setbacks, facade, reach)
                       if ref["pixels"] and not reference_id else None)
            if setback:
                key = setback.key
            elif reference_id:
                key = key_of.get(reference_id, facade)
            else:
                # Tapped points: on the facade's wall whose line runs nearest them.
                key = (walls_module.nearest_wall_key(ref["pixels"], lines, key_of, facade, reach)
                       if ref["pixels"] else None) or facade
        if key not in wall_facade:
            wall_facade[key] = facade
            normals[key] = prep.normals.get(facade) if facade in prep.normals else camera.wall_normal(facade)
        normal = normals.get(key)
        pixels = ref["pixels"]
        # The wall stands in front of the camera: its offset takes the sign that puts the measured
        # points there. (The wrong sign mirrors the house through the lens: lengths survive, but
        # up and down swap, and every rule asking which end of a line is the top goes wrong.)
        sign = 1.0
        if normal is not None and pixels:
            sign = 1.0 if float(normal @ camera.ray(pixels[0])) > 0 else -1.0
        unit = [_on_plane(camera, p, normal, sign) for p in pixels] if normal is not None and pixels else None
        solution.reference = {"wall": key, "facade": facade, "how": how}
        trace.step("3_measurement", facade=facade, wall=key, how=how, wall_normal_found=normal is not None, sign=sign)
        if unit is None or any(p is None for p in unit) or _length(unit) <= 0:
            solution.needs = "measurement_wall"
            trace.step("3_measurement", needs="measurement_wall")
        else:
            measured_plane = sign * ref["length_in"] / _length(unit)
            walls[key] = {"distance_in": measured_plane + sign * (EAVE_OVERHANG_IN if reference_on_eave else 0.0),
                          "source": "measured", "facade": facade}
            measured = key
            trace.step("3_measurement", length_at_unit_distance=_length(unit), measured_plane_in=measured_plane,
                       wall_distance_in=walls[key]["distance_in"],
                       overhang_added_in=EAVE_OVERHANG_IN if reference_on_eave else 0.0)

    if reference_id and measured:
        # The line the salesperson measured is on the wall they measured it against, whatever its
        # own rule would say: the scale was taken from it there, so it reads exactly the typed length.
        wall_of[reference_id] = measured
    for line_id, key in wall_of.items():
        trace.line(line_id, wall=key)

    # 5. Distances: through shared corners and through roof faces, until nothing new is reached.
    rejected: set = set()
    links = []
    changed = bool(walls)
    while changed:
        changed = False
        candidates: dict[str, list[float]] = {}
        through_pitch: dict[str, bool] = {}
        for a in lines:
            known = wall_of.get(a.id)
            if known not in walls:
                continue
            for b in lines:
                other = wall_of.get(b.id)
                if other is None or other in walls or normals.get(other) is None:
                    continue
                if wall_facade.get(other) == wall_facade.get(known):
                    continue    # parallel walls share no corner
                for end in (a.points[0], a.points[-1]):
                    for tip in (b.points[0], b.points[-1]):
                        if _ends_meet(end, tip, reach):
                            # The corner is on a's own plane, and on b's; b's wall is b's overhang
                            # behind that.
                            corner = [(end[0] + tip[0]) / 2, (end[1] + tip[1]) / 2]
                            if abs(float(normals[known] @ camera.ray(corner))) < GRAZING_MIN:
                                continue    # the known wall seen edge-on there: the corner's depth is noise
                            point = _on_plane(camera, corner, normals[known],
                                              _nearer(walls[known]["distance_in"], _overhang(a)))
                            if point is not None:
                                value = float(normals[other] @ point)
                                candidates.setdefault(other, []).append(value + float(np.sign(value)) * _overhang(b))
                                through_pitch[other] = through_pitch.get(other, False) or bool(walls[known].get("via_pitch"))
                                trace.add("4_walls", "corners", {
                                    "from_line": a.id, "from_wall": known, "to_line": b.id, "to_wall": other,
                                    "corner_px": [round(corner[0], 1), round(corner[1], 1)],
                                    "to_wall_distance_in": round(candidates[other][-1], 1)})
        for other, values in candidates.items():
            # Every point of a wall in view lies on one side of the camera. Corners that put the
            # wall through the camera, or on both sides of it, are corners of some other wall
            # (a misread line), and a median of them is a plane through the lens.
            values = [v for v in values if abs(v) > 1.0]
            if not values or (min(values) < 0 < max(values)):
                if other not in rejected:
                    rejected.add(other)
                    trace.add("4_walls", "rejected", {"wall": other, "values": [round(v, 1) for v in values],
                                                      "why": "no corner off the lens" if not values
                                                      else "corners on both sides of the camera"})
                continue
            walls[other] = {"distance_in": float(np.median(values)), "source": "corner",
                            "facade": wall_facade[other], "via_pitch": through_pitch.get(other, False)}
            changed = True
        for setback in setbacks:
            link = _link(camera, setback, walls, normals, by_id, up, angle, wall_of.get(setback.eave))
            if link:
                links.append(link)
                changed = True
        # A gable standing on its own wall's eave: a rake's foot on an eave of the same direction
        # puts the gable end in that eave's wall, at its distance. Parallel walls share no corner,
        # so without this a cross gable flush with the front wall, and every wall reached only
        # through it, stayed unmeasured (7d231dc3: the front gables, then the side gable beyond).
        for rake_id, key in gable_key.items():
            if key in walls or normals.get(key) is None:
                continue
            _, foot = _top_and_foot(by_id[rake_id])
            for other in lines:
                other_key = wall_of.get(other.id)
                if (other.edge not in WALL_EDGES or other_key not in walls
                        or wall_facade.get(other_key) != wall_facade.get(key)):
                    continue
                if _to_polyline(foot, other.points) <= reach:
                    walls[key] = {"distance_in": walls[other_key]["distance_in"], "source": "corner",
                                  "facade": wall_facade[key], "via_pitch": bool(walls[other_key].get("via_pitch"))}
                    links.append({"from": other_key, "to": key, "distance_in": round(walls[key]["distance_in"], 1),
                                  "why": "gable on its eave"})
                    changed = True
                    break
        if not changed:
            # A run of a facade no corner reaches is taken in line with a run of that facade that
            # is known (the measured one first): the old one-plane-per-side reading, and said so.
            # Only once nothing else moves, so a corner always wins over this.
            for key, facade in split_of.items():
                for k in (key, facade):
                    if k in walls:
                        continue
                    siblings = [s for s in (measured, facade, *split_of)
                                if s and s != k and s in walls and wall_facade.get(s) == facade]
                    if siblings:
                        source = walls[siblings[0]]
                        walls[k] = {"distance_in": source["distance_in"], "source": "same plane", "facade": facade,
                                    "via_pitch": bool(source.get("via_pitch")), "same_as": siblings[0]}
                        trace.add("4_walls", "same_plane", {"wall": k, "as": siblings[0]})
                        changed = True
    trace.step("4_walls", walls={k: dict(w) for k, w in walls.items()}, links=links,
               unreached=[k for k in wall_facade if k not in walls])

    # 6. Placement.
    focal_note = prep.focal_note
    overhang_note = f"eave overhang assumed to be {_feet(EAVE_OVERHANG_IN)}"
    link_note = f"its wall's depth comes through the roof face between it and the next wall, at {pitch_words}"

    def on_wall(line: Line, key: str, method: str, plane: str) -> Placed:
        normal = normals[key]
        distance = _nearer(walls[key]["distance_in"], _overhang(line))
        points = [_on_plane(camera, p, normal, distance) for p in line.points]
        if any(p is None for p in points):
            return Placed(line.id, needs="edge_on", side=facade_of.get(line.id), side_how=side_how.get(line.id))
        notes, confidence = [], EXACT
        if key != measured and focal_note:
            notes.append(focal_note)
            confidence = ASSUMED
        if walls[key].get("via_pitch"):
            notes.append(link_note)
            confidence = _worst(confidence, pitch_sure)
        if walls[key].get("source") == "same plane":
            notes.append(f"taken in line with the {wall_facade.get(key, key)} wall it does not join: "
                         "no corner gives its own depth")
            confidence = _worst(confidence, STATED)
        if side_how.get(line.id) == "borrowed":
            notes.append("side taken from the wall line next to it: its own direction is unclear")
            confidence = _worst(confidence, STATED)
        # The overhang moves a line only relative to a reference on the other kind of plane: an
        # eave against a wall measurement, or a wall line against an eave measurement.
        if (line.edge == "eave") != reference_on_eave:
            notes.append(overhang_note)
        if side_how.get(line.id) == "chosen":
            notes.append("side chosen by consistency: its direction fits more than one wall")
            confidence = _worst(confidence, STATED)
        return Placed(line.id, points, _length(points), plane, method, assumption="; ".join(notes) or None,
                      confidence=confidence, wall=key, side=facade_of.get(line.id), side_how=side_how.get(line.id))

    placed: dict[str, Placed] = {}
    for line in lines:
        key = wall_of.get(line.id)
        if key is None or (line.edge not in WALL_EDGES and line.id != reference_id):
            continue
        if key not in walls:
            placed[line.id] = Placed(line.id, needs="measurement", side=facade_of.get(line.id),
                                     side_how=side_how.get(line.id))
            continue
        where = wall_facade.get(key, key) + (" set-back wall" if "+" in key else " wall")
        if line.id == reference_id:
            placed[line.id] = on_wall(line, key, "measured", f"{where}, your measurement")
            continue
        grazing = _grazing(camera, line.points, normals[key])
        if grazing < GRAZING_MIN:
            # Its wall runs nearly along the line of sight (toward a vanishing point in the photo):
            # a pixel there is feet of depth.
            trace.line(line.id, wall_grazing_deg=round(float(np.degrees(np.arcsin(min(1.0, grazing)))), 1))
            placed[line.id] = Placed(line.id, needs="edge_on", wall=key, side=facade_of.get(line.id),
                                     side_how=side_how.get(line.id))
            continue
        placed[line.id] = on_wall(line, key, "wall", where)

    eaves = [(line, placed[line.id]) for line in lines
             if line.edge == "eave" and line.id in placed and placed[line.id].points3d]

    def eave_at_foot(line: Line) -> tuple[Line, Placed] | None:
        """The placed eave whose end is at the line's lower end, if any."""
        _, foot = _top_and_foot(line)
        return next(((e, done) for e, done in eaves
                     if any(_ends_meet(foot, tip, reach) for tip in (e.points[0], e.points[-1]))), None)

    face_level: dict[str, np.ndarray] = {}   # a line put on a roof face -> that face's level direction

    def on_face(line: Line, level: np.ndarray, through: np.ndarray, method: str, plane: str, why: str | None,
                sure: str | None, back: bool = False, source: str | None = None) -> Placed:
        normal, offset = _roof_face(level, through, up, angle, back)
        points = [_on_plane(camera, p, normal, offset) for p in line.points]
        if any(p is None for p in points):
            return Placed(line.id, needs="edge_on")
        notes = [n for n in (why, f"depth from {pitch_words}", focal_note) if n]
        trace.line(line.id, face="back" if back else "front", face_from=source)
        face_level[line.id] = level
        return Placed(line.id, points, _length(points), plane, method, assumption="; ".join(notes),
                      confidence=_worst(pitch_sure, sure))

    def from_eave(line: Line, eave: tuple[Line, Placed], plane: str, why: str | None) -> Placed:
        done = eave[1]
        return on_face(line, done.points3d[-1] - done.points3d[0], (done.points3d[0] + done.points3d[-1]) / 2,
                       "pitch", plane, why, done.confidence, source=eave[0].id)

    def gable_through_foot(line: Line, gable: str) -> None:
        """A gable wall no corner reached (its rake was not a rake then): through the corner at the
        rake's foot, with the placed wall line there."""
        values, through_pitch = [], False
        for end in (line.points[0], line.points[-1]):
            for other in lines:
                done = placed.get(other.id)
                if other.edge not in WALL_EDGES or not done or not done.points3d or done.wall not in walls:
                    continue
                for tip in (other.points[0], other.points[-1]):
                    if not _ends_meet(end, tip, reach):
                        continue
                    corner = [(end[0] + tip[0]) / 2, (end[1] + tip[1]) / 2]
                    point = _on_plane(camera, corner, normals[done.wall],
                                      _nearer(walls[done.wall]["distance_in"], _overhang(other)))
                    if point is not None and abs(float(normals[gable] @ point)) > 1.0:
                        values.append(float(normals[gable] @ point))
                        through_pitch = through_pitch or bool(walls[done.wall].get("via_pitch"))
        if values and not min(values) < 0 < max(values):
            walls[gable] = {"distance_in": float(np.median(values)), "source": "corner",
                            "facade": wall_facade.get(gable, gable), "via_pitch": through_pitch}

    pending: list[tuple[Line, Placed | None, str, str | None, str]] = []

    def place_rake(line: Line, key: str | None, how: str | None, prefix: str = "") -> None:
        """On its gable wall when the camera sees that well; else on the roof face rising from the
        eave at its foot; else it waits for the far-slope pass below."""
        gable = None
        if key is not None and key not in walls and normals.get(key) is not None:
            gable_through_foot(line, key)
        if key is not None and key in walls:
            wall_of[line.id] = key
            gable = on_wall(line, key, "gable", f"{wall_facade.get(key, key)} gable wall")
            grazing = _grazing(camera, line.points, normals[key])
            trace.line(line.id, gable_grazing_deg=round(float(np.degrees(np.arcsin(min(1.0, grazing)))), 1))
            inferred = {"elimination": "its gable wall is inferred: the side other than its eave's",
                        "plane": "its gable wall is inferred: the only side on which it has a believable pitch",
                        "given": "its gable wall is the side detection gave it"}.get(how or "")
            if inferred and gable.points3d:
                gable.confidence = _worst(gable.confidence, STATED)
                gable.assumption = "; ".join(n for n in (prefix.strip("; ") or None, inferred,
                                                         gable.assumption) if n)
            elif prefix and gable.points3d:
                gable.assumption = "; ".join(n for n in (prefix.strip("; "), gable.assumption) if n)
            if gable.points3d and grazing >= GRAZING_MIN:
                placed[line.id] = gable
                return
        foot = eave_at_foot(line)
        why = prefix + ("its gable wall is seen nearly edge-on" if gable is not None
                        else "no gable wall in view" if key is None else "its gable wall's distance is unknown")
        if foot is not None:
            face = from_eave(line, foot, "roof face from the eave at its foot", why)
            if face.points3d:
                placed[line.id] = face
                return
        pending.append((line, gable, why, how, prefix))

    # Rakes: on the gable wall when the camera sees it, else on the roof face from their eave.
    for line in lines:
        if line.edge == "rake" and line.id not in as_hip and line.id not in placed:
            place_rake(line, wall_of.get(line.id), gable_how.get(line.id))

    # (b) hips from eave corners, and rakes read as hips.
    eave_ends = []
    for line in lines:
        done = placed.get(line.id)
        if line.edge in WALL_EDGES and done and done.points3d and len(done.points3d) >= 2:
            eave_ends += [(line.points[0], done.points3d[0], done.points3d[-1]),
                          (line.points[-1], done.points3d[-1], done.points3d[0])]
    for line in lines:
        if (line.edge != "hip" and line.id not in as_hip) or line.id in placed:
            continue
        for i, end in enumerate((line.points[0], line.points[-1])):
            corners = sorted((c for c in eave_ends if _ends_meet(end, c[0], reach)),
                             key=lambda c: np.hypot(c[0][0] - end[0], c[0][1] - end[1]))
            if not corners:
                continue
            base = corners[0][1]
            alongs: list[np.ndarray] = []
            for _, here, there in corners:
                along = _horizontal(there - here, up)
                if np.linalg.norm(along) < 1e-9:
                    continue
                along = _unit(along)
                if all(abs(float(along @ a)) < 0.9 for a in alongs):
                    alongs.append(along)
            if not alongs:
                continue
            bisector, how = _hip_bisector(alongs, base, up)
            normal = _unit(np.cross(bisector, up))
            top_pixel = line.points[-1] if i == 0 else line.points[0]
            top = _on_plane(camera, top_pixel, normal, float(normal @ base))
            if top is None:
                continue
            run = float(np.linalg.norm(_horizontal(top - base, up))) / np.sqrt(2)
            implied = 12.0 * float((top - base) @ up) / run if run > 1e-9 else float("inf")
            trace.line(line.id, hip_from_eave_corner_px=[round(float(v), 1) for v in end], hip_bisector=how,
                       hip_implied_pitch=round(implied, 2) if np.isfinite(implied) else None)
            plausible = PLAUSIBLE_PITCH[0] <= implied <= PLAUSIBLE_PITCH[1]
            if line.id in as_hip and not plausible:
                trace.line(line.id, hip_rejected="implied pitch implausible: back to the rake rule")
                break   # no believable hip either: it stays a rake, below
            if not plausible:
                # A hip that would slope down, or rise like a wall, is not this hip: a misread line
                # (often a receding eave). Refused rather than shown with a wrong length.
                trace.line(line.id, hip_rejected="implied pitch implausible: not placed")
                placed[line.id] = Placed(line.id, needs="unreliable")
                break
            points = [base, top] if i == 0 else [top, base]
            placed[line.id] = Placed(
                line.id, points, _length(points), "over the corner bisector", "hip",
                assumption=(HIP_WHY[as_hip[line.id]] if line.id in as_hip else "")
                + f"adjacent roof faces share one pitch (implies {implied:.1f}/12)"
                + (f"; {focal_note}" if focal_note else ""),
                confidence=_worst(STATED, corners_confidence(placed, lines, end, reach)))
            break

    # A rake read as a hip that the hip rule could not place goes back to the rake rule.
    for line in lines:
        if line.id in as_hip and line.id not in placed:
            gable, how = _gable_facade(camera, line, lines, reach, prep.tolerance, prep.facades, facade_of,
                                       prep.normals, up)
            key = None
            if gable is not None and prep.normals.get(gable) is not None:
                top, _ = _top_and_foot(line)
                key = next((k for rake_id, k in gable_key.items() if wall_facade.get(k) == gable
                            and _ends_meet(top, _top_and_foot(by_id[rake_id])[0], reach)), None)
                if key is None:
                    taken = {k for k in gable_key.values() if wall_facade.get(k) == gable}
                    key = f"{gable}#{len(taken) + 1}"
                register_gable(line.id, gable, key)
            place_rake(line, key, how, "not a believable hip; ")

    # A rake that waited because its gable wall had no distance yet, which a later rake of the
    # same gable then gave it (through the eave at that rake's own foot): on that wall now.
    waiting, pending = pending, []
    for line, gable, why, how, prefix in waiting:
        key = wall_of.get(line.id)
        if line.id not in placed and gable is None and key in walls:
            trace.line(line.id, gable_wall_retried=key)
            place_rake(line, key, how, prefix)
        else:
            pending.append((line, gable, why, how, prefix))

    # The far slope of a gable: a rake with no eave at its foot whose top is a peak already placed
    # on a roof face lies on the face beyond the ridge. Else its gable wall, edge-on, if it has one.
    for line, gable, why, _, _ in pending:
        if line.id in placed:
            continue
        top, _ = _top_and_foot(line)
        partner = next((other for other in lines if other is not line and other.id in face_level
                        and placed.get(other.id) and placed[other.id].points3d
                        and any(_ends_meet(top, tip, reach) for tip in (other.points[0], other.points[-1]))), None)
        if partner is not None:
            done = placed[partner.id]
            peak = max(done.points3d, key=lambda p: float(p @ up))
            face = on_face(line, face_level[partner.id], peak, "pitch", "roof face beyond the ridge",
                           "the far slope of the gable, from its peak", done.confidence, back=True, source=partner.id)
            if face.points3d:
                placed[line.id] = face
                continue
        if gable is not None and gable.points3d:
            gable.confidence = ASSUMED
            gable.assumption = "; ".join(n for n in ("its gable wall is seen nearly edge-on", gable.assumption) if n)
            placed[line.id] = gable
        elif wall_of.get(line.id) is not None:
            placed[line.id] = Placed(line.id, needs="measurement")

    # (a) ridges from gable peaks, rake tops and hip tops. As sure as the anchors.
    def anchored_ends() -> list[tuple[list[float], np.ndarray, str, str]]:
        out = []
        for line in lines:
            done = placed.get(line.id)
            if not done or not done.points3d or done.method == "override":
                continue
            top = max(range(len(done.points3d)), key=lambda i: float(done.points3d[i] @ up))
            if line.edge == "rake" and done.method in ("gable", "measured", "wall", "pitch"):
                peak = any(other is not line and other.edge == "rake" and
                           any(_ends_meet(line.points[top], tip, reach) for tip in (other.points[0], other.points[-1]))
                           for other in lines)
                if peak:
                    sure = _worst(done.confidence, EXACT if not focal_note else STATED)
                    out.append((line.points[top], done.points3d[top], "gable peak", sure))
                else:
                    out.append((line.points[top], done.points3d[top], "rake top", _worst(done.confidence, STATED)))
            elif line.edge == "hip" or done.method == "hip":
                out.append((line.points[top], done.points3d[top], "hip top", done.confidence or STATED))
        return out

    anchors = anchored_ends()
    trace.step("6_anchors", anchors=[{"px": [round(float(v), 1) for v in a[0]], "what": a[2], "sure": a[3]}
                                     for a in anchors])

    # Two anchors at one point (a gable's peak that a misread hip also ends at) can disagree in
    # depth: the surer one holds, a gable peak before a rake's top before a hip's.
    kind_rank = {"gable peak": 0, "rake top": 1, "hip top": 2}

    def anchor_at(pixel):
        found = [a for a in anchors if _ends_meet(pixel, a[0], reach)]
        return min(found, key=lambda a: (RANK.get(a[3], len(RANK)), kind_rank.get(a[2], len(kind_rank)),
                                         np.hypot(a[0][0] - pixel[0], a[0][1] - pixel[1]))) if found else None

    for line in lines:
        if line.edge != "ridge" or line.id in placed:
            continue
        ends = (anchor_at(line.points[0]), anchor_at(line.points[-1]))
        trace.line(line.id, ridge_anchors=[end[2] if end else None for end in ends])
        if ends[0] is not None and ends[1] is not None and ends[0][1] is not ends[1][1]:
            points = [ends[0][1], ends[1][1]]
            whats = sorted({ends[0][2], ends[1][2]})
            confidence = _worst(ends[0][3], ends[1][3])
            assumption = None if whats == ["gable peak"] else (
                "from a hip, which assumes one pitch" if "hip top" in whats else "from a rake's top")
            notes = [n for n in (assumption, focal_note) if n]
            placed[line.id] = Placed(line.id, points, _length(points), f"between the {' and the '.join(whats)}s"
                                     if len(whats) == 1 else f"between the {whats[0]} and the {whats[1]}", "ridge",
                                     assumption="; ".join(notes) or None, confidence=confidence)
            continue
        facade = facade_of.get(line.id)
        if facade not in prep.normals or not camera.horizontal or facade not in camera.facades:
            continue
        level = camera.direction(camera.horizontal[camera.facades.index(facade)])
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

    # (c0) a valley whose two ends lie on placed lines (a dormer's side eave above, the line where
    # its wall meets the roof below): between those two points. Both are known in 3-D, so no pitch
    # is assumed (6c910b84: a 110 px valley read 20 ft off the nearest eave's face while the
    # dormer's 290 px rakes read 17 ft).
    wall_lines = [(other, placed[other.id]) for other in lines
                  if other.edge in WALL_EDGES + ("rake",) and other.id in placed
                  and placed[other.id].points3d and placed[other.id].method != "override"]

    def on_placed(pixel, line: Line):
        """(point, the placed line it lies on) for a pixel on or at the end of a placed wall line."""
        best = None
        for other, done in wall_lines:
            if other is line:
                continue
            # At a line's end, or along an eave or a junction (a valley starts on them); never
            # part-way along a rake, which a valley only passes near.
            ends = min(float(np.hypot(pixel[0] - q[0], pixel[1] - q[1])) for q in (other.points[0], other.points[-1]))
            gap = ends if other.edge == "rake" else min(ends, _to_polyline(pixel, other.points))
            if gap <= reach and (best is None or gap < best[0]):
                best = (gap, other, done)
        if best is None:
            return None
        _, other, done = best
        point = _closest_on_line(camera, pixel, done.points3d[0], done.points3d[-1] - done.points3d[0])
        return point, other, done

    for line in lines:
        if line.id in placed or line.edge != "valley":
            continue
        ends = (on_placed(line.points[0], line), on_placed(line.points[-1], line))
        if ends[0] is None or ends[1] is None or ends[0][1] is ends[1][1]:
            continue
        points = [ends[0][0], ends[1][0]]
        if any(float(q[2]) <= 0 for q in points) or _length(points) <= 0:
            continue
        notes = [n for n in (focal_note,) if n]
        notes.insert(0, "its ends taken where it meets those lines")
        placed[line.id] = Placed(line.id, points, _length(points),
                                 f"between the {ends[0][1].edge} and the {ends[1][1].edge} it joins", "valley",
                                 assumption="; ".join(notes),
                                 confidence=_worst(STATED, ends[0][2].confidence, ends[1][2].confidence))
        trace.line(line.id, valley_between=[ends[0][1].id, ends[1][1].id])

    # (c) the rest of the ridges, hips and valleys: the face rising from the nearest eave.
    for line in lines:
        if line.id in placed or line.edge not in ("ridge", "hip", "valley"):
            continue
        if not eaves:
            placed[line.id] = Placed(line.id, needs="measurement" if not walls else "anchor")
            continue
        middle = np.mean(np.asarray(line.points, float), axis=0)

        def distance(item) -> float:
            a, b = np.asarray(item[0].points[0], float), np.asarray(item[0].points[-1], float)
            t = np.clip(float((middle - a) @ (b - a)) / max(float((b - a) @ (b - a)), 1e-9), 0, 1)
            return float(np.linalg.norm(middle - (a + t * (b - a))))

        # A ridge runs parallel to the eave its face rises from: with a side, only that side's eaves
        # (824fd558: a front-running ridge hung from the side gable's eave came out tilted and 22 ft).
        same = [item for item in eaves if line.edge == "ridge" and facade_of.get(line.id)
                and facade_of.get(item[0].id) == facade_of.get(line.id)]
        # A valley or hip lies on a roof face that comes down to an eave below it, not to a
        # dormer's side eave beside it (6c910b84).
        lowest = max(p[1] for p in line.points)
        below = [item for item in eaves if line.edge in ("valley", "hip")
                 and min(q[1] for q in item[0].points) > lowest]
        placed[line.id] = from_eave(line, min(same or below or eaves, key=distance), "roof face from the nearest eave", None)

    # Nothing of the house lies behind the camera, or far beyond its farthest wall: a point there
    # is a ray meeting a plane at a grazing angle, not a place on the house.
    farthest = max((abs(float(w["distance_in"])) for w in walls.values()), default=0.0)
    if farthest > 0:
        limit = max(2.0 * farthest, farthest + DEPTH_SLACK_IN)
        for line_id, done in list(placed.items()):
            if not done.points3d or done.method == "measured":
                continue
            if any(float(p[2]) <= 0 or float(np.linalg.norm(p)) > limit for p in done.points3d):
                trace.line(line_id, refused=f"placed beyond the house (over {limit / 12:.0f} ft from the camera)",
                           refused_length_in=round(done.length_in, 1) if done.length_in else None)
                placed[line_id] = Placed(line_id, needs="unreliable", wall=done.wall, side=done.side,
                                         side_how=done.side_how)

    # Anything left, and every line the salesperson overrode.
    for line in lines:
        # Only a midpoint or a length is a line override; other keys belong to areas.
        override = {k: v for k, v in (prep.overrides.get(line.id) or {}).items() if k in ("mid", "length_in")}
        done = placed.get(line.id)
        if override:
            length = override.get("length_in") or (done.length_in if done and done.points3d else None)
            placed[line.id] = Placed(line.id, done.points3d if done else None, length, "set by hand", "override",
                                     assumption="midpoint placed by hand",
                                     confidence=STATED if length else None,
                                     needs=None if length else "length",
                                     warning=_questioned(override.get("length_in"),
                                                         done.length_in if done and done.points3d
                                                         else _wall_reading(camera, walls, normals, placed, by_id, line)))
        elif done is None:
            reason = "measurement" if not walls else ("side" if line.edge in LEVEL and not facade_of.get(line.id)
                                                      else "override")
            placed[line.id] = Placed(line.id, needs=reason)
        if line.id in placed and placed[line.id].side is None and line.id in facade_of:
            placed[line.id].side, placed[line.id].side_how = facade_of.get(line.id), side_how.get(line.id)
    solution.lines = placed

    # 7. Checks against what the solve did not produce.
    features = []
    for feature in prep.features:
        if not feature.get("facade"):
            continue
        span = [feature["left"], feature["right"]]
        setback = walls_module.setback_of_points(span, lines, setbacks, feature["facade"], reach)
        wall = setback.key if setback else (walls_module.nearest_wall_key(span, lines, key_of, feature["facade"], reach)
                                            or feature["facade"])
        features.append({**feature, "wall": wall})
    context = checks_module.Context(camera, lines, placed, walls, wall_of, normals, up, measured, reach,
                                    _overhang, features, len(rejected))
    solution.penalty, solution.checks = checks_module.evaluate(context)
    undecided = [line_id for line_id, how in side_how.items() if how == "ambiguous"]
    if undecided:
        solution.checks.append({"kind": "side", "lines": undecided,
                                "message": "The side of the house some lines are on is unclear from the photo; "
                                           "check their side."})
    solution.warning = next((c["message"] for c in solution.checks if c["kind"] in ("plausibility", "size")), None)
    trace.step("8_checks", penalty=round(solution.penalty, 3), checks=solution.checks,
               features=[{k: f.get(k) for k in ("id", "kind", "label", "wall", "measured", "side_assignment")}
                         for f in features])
    for line_id, done in placed.items():
        trace.line(line_id, **done.to_dict())
    trace.step("9_result", needs=solution.needs, warning=solution.warning, pitch=solution.pitch,
               walls={k: dict(w) for k, w in walls.items()}, reference=solution.reference)
    return solution


ESTIMATE_NEIGHBOURS = 3          # placed lines whose scale an unplaced line borrows
ESTIMATE_SAME_DIRECTION_DEG = 30.0


def _pixel_length(points) -> float:
    return float(sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:])))


def _estimate_from_neighbours(lines: list[Line], solution: "Solution", reach: float) -> None:
    """A length for every line no rule could place (no wall reached, an unknown edge, a scrap no
    corner joins) or whose own length was too unsure to show (_uncertainty), read off the parts of
    the house around it: the inches per pixel of the placed lines nearest it in the photo, those
    running the same way and of its own kind first (a rake beside a rake foreshortens as it does).
    Marked assumed, and says which lines it came from, so a hand-set length or a reference on that
    part still replaces it. Runs after _uncertainty, so it reads only lengths that held."""
    if not solution.walls:
        return
    placed = solution.lines
    sources = []
    for other in lines:
        done = placed.get(other.id)
        if (done is None or not done.points3d or not done.length_in or done.method in ("override", "estimate")
                or len(other.points) < 2):
            continue
        pixels = _pixel_length(other.points)
        if pixels >= 1.0:
            sources.append((other, done, done.length_in / pixels))
    if not sources:
        return
    for line in lines:
        done = placed.get(line.id)
        if done is not None and (done.length_in is not None or done.method == "override"):
            continue
        pixels = _pixel_length(line.points)
        if pixels < 1.0:
            continue
        a, b = np.asarray(line.points[0], float), np.asarray(line.points[-1], float)
        direction = _unit(b - a)
        middle = (a + b) / 2
        scored = []
        for other, other_done, ratio in sources:
            gap = min(_to_polyline(middle, other.points), _to_polyline(other.points[0], line.points),
                      _to_polyline(other.points[-1], line.points))
            o = _unit(np.subtract(other.points[-1], other.points[0], dtype=float))
            turn = float(np.degrees(np.arccos(min(1.0, abs(float(direction @ o))))))
            score = (gap + reach) * (1.0 + turn / ESTIMATE_SAME_DIRECTION_DEG) * (1.0 if other.edge == line.edge else 1.5)
            scored.append((score, other, other_done, ratio))
        scored.sort(key=lambda item: item[0])
        chosen = scored[:ESTIMATE_NEIGHBOURS]
        weights = np.array([1.0 / item[0] for item in chosen])
        ratio = float(np.sum(weights * np.array([item[3] for item in chosen])) / np.sum(weights))
        names = ", ".join(f"the {item[1].edge or 'line'} at ({item[1].points[0][0]:.0f}, {item[1].points[0][1]:.0f}) "
                          f"({_feet(item[2].length_in)})" for item in chosen)
        unsure = done is not None and done.needs == "unreliable"
        why = (f"its own geometry is unsure (it moves {done.spread:.0%} under the photo's own errors)"
               if unsure and done.spread is not None and np.isfinite(done.spread)
               else "its own geometry is unsure" if unsure else "nothing in the photo places it")
        placed[line.id] = Placed(
            line.id, None, ratio * pixels, "estimated from the parts of the house around it", "estimate",
            assumption=f"{why}, so its length is read at the scale of {names}; "
                       "set it by hand, or measure that part, for a firm figure",
            confidence=ASSUMED, needs=None, wall=done.wall if done else None,
            side=done.side if done else None, side_how=done.side_how if done else None,
            spread=done.spread if done else None)


def corners_confidence(placed: dict, lines: list[Line], end, reach: float) -> str | None:
    """The worst confidence among the placed wall lines meeting at a corner: a hip is no surer
    than the eaves it rises from."""
    found = [placed[line.id].confidence for line in lines
             if line.edge in WALL_EDGES and line.id in placed and placed[line.id].points3d
             and any(_ends_meet(end, tip, reach) for tip in (line.points[0], line.points[-1]))]
    return _worst(*found)


def _link(camera: Camera, setback: walls_module.SetBack, walls: dict, normals: dict, by_id: dict,
          up: np.ndarray, angle: float, main: str | None = None) -> dict | None:
    """A set-back wall's distance from the wall of the eave below it (`main`, by default its
    facade's main wall), or the other way round, through the roof face rising from that eave to
    the junction at its foot."""
    eave, junction = by_id.get(setback.eave), by_id.get(setback.junction)
    normal = normals.get(setback.facade)
    if eave is None or junction is None or normal is None:
        return None
    main, high = main or setback.facade, setback.key
    if main in walls and high not in walls:
        known = walls[main]["distance_in"]
        bottom = [_on_plane(camera, p, normal, _nearer(known, _overhang(eave))) for p in eave.points]
        if any(p is None for p in bottom):
            return None
        face_normal, offset = _roof_face(bottom[-1] - bottom[0], (bottom[0] + bottom[-1]) / 2, up, angle)
        top = [_on_plane(camera, p, face_normal, offset) for p in junction.points]
        if any(p is None for p in top):
            return None
        found = float(np.mean([normal @ p for p in top]))
        if abs(found) <= 1.0 or np.sign(found) != np.sign(known):
            return None
        walls[high] = {"distance_in": found, "source": "roof face", "facade": setback.facade, "via_pitch": True}
        return {"from": main, "to": high, "distance_in": round(found, 1)}
    if high in walls and main not in walls:
        known = walls[high]["distance_in"]
        top = [_on_plane(camera, p, normal, known) for p in junction.points]
        if any(p is None for p in top):
            return None
        face_normal, offset = _roof_face(top[-1] - top[0], (top[0] + top[-1]) / 2, up, angle)
        bottom = [_on_plane(camera, p, face_normal, offset) for p in eave.points]
        if any(p is None for p in bottom):
            return None
        plane = float(np.mean([normal @ p for p in bottom]))
        found = plane + float(np.sign(plane)) * _overhang(eave)
        if abs(found) <= 1.0 or np.sign(found) != np.sign(known):
            return None
        walls[main] = {"distance_in": found, "source": "roof face", "facade": setback.facade, "via_pitch": True}
        return {"from": high, "to": main, "distance_in": round(found, 1)}
    return None
