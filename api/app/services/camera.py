"""The pinhole camera behind a photo, and the vertical walls of the house (M3: scale, perspective).

A photo is a pinhole projection x ~ K·X, with K = [[f, 0, cx], [0, f, cy], [0, 0, 1]] and the
principal point at the photo's centre. A vanishing point v is the image of the 3-D direction
K⁻¹·v. A facade is a vertical plane: it holds its own level direction (its horizontal vanishing
point) and the vertical (the vertical vanishing point), so its normal is their cross product.
How far a wall is from the camera is the one thing a photo cannot say; a known length on it
fixes that, and a pixel on the wall is then its camera ray meeting the plane.

The focal length, in order. EXIF is usually missing (the stored copy is a PNG, and phone
browsers strip it before upload), so the vanishing points are the normal path, not a fallback:

  1. "vp": two vanishing points of perpendicular directions, v1 and v2, give
     f² = −(v1 − c)·(v2 − c). Two facades' level directions, or one and the vertical. A pair is
     used only when it is well conditioned: turning either point by FOCAL_JITTER_DEG about the
     centre, or moving it nearer or further as the photo's lines aiming at it would if each were
     off by FOCAL_JITTER_DEG, must move f by less than FOCAL_SPREAD_MAX. A front-on facade's
     point lies near infinity, and then f is decided by a fraction of a degree of the eaves' tilt
     (a 114 mm lens read off a 0.18° tilt); a level camera's vertical point does too (an 80 mm
     lens read off uprights converging by half a degree across the photo). Such a pair is noise
     and is dropped (focal_checks.vp_rejected).
  2. "exif": the upload's 35 mm-equivalent focal, f = f35 / 36 × the long edge. Used when the
     vanishing points give none, and recorded beside theirs as a cross-check when they do.
  3. "assumed": a typical phone lens, and labelled so. Expect it on level, front-on photos,
     where the vanishing points go to infinity. The wall then faces the camera and its scale is
     uniform along it, so the focal length does not change spacing on that wall; it matters
     only for depth (other walls, hips, trunks), and those results carry the label.

The vanishing points themselves come from detection (services/fusion.py: the photo's, refit to
the target house), stored with the photo, so everything here is recomputed free after any edit.
The vertical refit and the search for a missed second facade use only the target house's own
segments (Scene.on_house) when detection knew the house: a neighbour's roof, a fence or a
driveway must not build a wall of this house.
"""

from dataclasses import dataclass, field

import numpy as np

from app.services import fusion, partition

ASSUMED_FOCAL_35MM = 26.0         # a typical phone main camera, 35 mm equivalent
FILM_WIDTH_MM = 36.0              # the 35 mm frame's long side, which the photo's long edge maps to
PLAUSIBLE_FOCAL_35MM = (12.0, 200.0)  # a VP estimate outside this is two VPs that are not perpendicular
EXIF_AGREES = 0.15                # relative: EXIF and the VP estimate this close confirm each other
AT_INFINITY = 1e-9
# A "vertical" vanishing point more than this far off the photo's own vertical, seen from its
# centre, is not the vertical: a steeply foreshortened wall's level lines taken for uprights.
VERTICAL_MAX_DEG = 25.0
# ... nor one nearer the centre than this share of the long edge. The vertical lies f / tan(tilt)
# from the centre, so this near it means a camera tilted 55 degrees (26 mm lens) or 38 (14 mm):
# no hand-held house photo. A point there is on the horizon, where the photo's tilt puts it
# (a side wall receding straight from a front-on camera), and its lines run up the photo only
# because it is close (9fd3d8e1: (499, 55), 286 px from the centre of a 1024 px photo, while a
# "facade" point 9688 px straight down was the real vertical).
VERTICAL_MIN_REACH = 0.5
# Finding the facade detection missed: a segment agrees with a vanishing point within this; the
# found point must hold this share of the segments' length that no other point explains, and at
# least this many of them.
AGREE_DEG = 2.0
SECOND_SHARE = 0.25
SECOND_MIN_SEGMENTS = 8
FOCAL_SWEEP_35MM = np.geomspace(14.0, 120.0, 160)
FOCAL_JITTER_DEG = 0.5            # how far a vanishing point's direction is turned to test its pair
FOCAL_SPREAD_MAX = 0.15           # relative change in f that jitter may cause for the pair to count
UPRIGHT_DEG = 15.0                 # segments this close to the photo's vertical are fitted for the vertical
UPRIGHT_MIN_SEGMENTS = 3


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > AT_INFINITY else v


def _pixel_vp(vp: np.ndarray, scene: partition.Scene) -> list[float]:
    """A scene vanishing point (homogeneous, normalised) in photo pixels: [x, y, 1], or a
    direction [dx, dy, 0] when it lies at infinity."""
    s, (cx, cy) = scene.scale, scene.centre
    x, y, w = float(vp[0]) * s + cx * float(vp[2]), float(vp[1]) * s + cy * float(vp[2]), float(vp[2])
    if abs(w) < AT_INFINITY:
        dx, dy = _unit(np.array([x, y]))
        return [float(dx), float(dy), 0.0]
    return [x / w, y / w, 1.0]


def _focal_from_pair(a: list[float], b: list[float], centre: np.ndarray) -> float | None:
    """f from two finite vanishing points of perpendicular directions, or None."""
    if a[2] == 0.0 or b[2] == 0.0:
        return None
    square = -float(np.dot(np.array(a[:2]) - centre, np.array(b[:2]) - centre))
    return float(np.sqrt(square)) if square > 0 else None


def _turned(vp: list[float], centre: np.ndarray, degrees: float) -> list[float]:
    """A finite vanishing point turned about the photo's centre."""
    angle = np.radians(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x, y = rotation @ (np.array(vp[:2], float) - centre) + centre
    return [float(x), float(y), 1.0]


def _moved(vp: list[float], centre: np.ndarray, degrees: float) -> list[float] | None:
    """A finite vanishing point moved nearer the centre or further out, as it would be if the
    photo's lines aiming at it from either edge (half the long edge off its axis) were each off
    by `degrees`. A far point moves a lot: its lines are nearly parallel, and their slight
    convergence is all that places it. None when the lines would no longer meet on that side."""
    offset = np.array(vp[:2], float) - centre
    distance = float(np.linalg.norm(offset))
    if distance < AT_INFINITY:
        return list(vp)
    half = float(np.max(centre))
    angle = float(np.arctan2(half, distance)) + np.radians(degrees)
    if angle <= 0 or angle >= np.pi / 2:
        return None
    x, y = centre + offset / distance * (half / np.tan(angle))
    return [float(x), float(y), 1.0]


def focal_spread(a: list[float], b: list[float], centre: np.ndarray) -> float:
    """How far f moves, relative, when either vanishing point is off by FOCAL_JITTER_DEG, turned
    about the centre or moved along its line from it: small for a well-placed pair, huge (or
    inf) when a point is near infinity."""
    base = _focal_from_pair(a, b, centre)
    if base is None:
        return float("inf")

    def variants(vp):
        return ([_turned(vp, centre, d) for d in (-FOCAL_JITTER_DEG, FOCAL_JITTER_DEG)]
                + [_moved(vp, centre, d) for d in (-FOCAL_JITTER_DEG, FOCAL_JITTER_DEG)])

    worst = 0.0
    for moved_a in variants(a):
        for moved_b in variants(b):
            moved = (_focal_from_pair(moved_a, moved_b, centre)
                     if moved_a is not None and moved_b is not None else None)
            if moved is None:
                return float("inf")
            worst = max(worst, abs(moved - base) / base)
    return worst


def _to_35mm(focal_px: float, long_edge: float) -> float:
    return focal_px * FILM_WIDTH_MM / long_edge


def _from_35mm(focal_35mm: float, long_edge: float) -> float:
    return focal_35mm / FILM_WIDTH_MM * long_edge


@dataclass
class Camera:
    width: int
    height: int
    focal: float                              # px
    focal_source: str                         # vp | exif | assumed
    vertical: list[float] | None = None       # [x, y, 1] px, or [dx, dy, 0] at infinity
    horizontal: list[list[float]] = field(default_factory=list)
    facades: list[str] = field(default_factory=list)   # per horizontal VP: front | left | right
    focal_checks: dict = field(default_factory=dict)   # focal_35mm per source that gave one

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.width / 2.0, self.height / 2.0])

    @property
    def K(self) -> np.ndarray:
        cx, cy = self.centre
        return np.array([[self.focal, 0.0, cx], [0.0, self.focal, cy], [0.0, 0.0, 1.0]])

    @property
    def focal_35mm(self) -> float:
        return _to_35mm(self.focal, max(self.width, self.height))

    def ray(self, pixel) -> np.ndarray:
        """The unit direction from the camera through a pixel."""
        return _unit(np.linalg.solve(self.K, np.array([float(pixel[0]), float(pixel[1]), 1.0])))

    def direction(self, vp: list[float]) -> np.ndarray:
        """The 3-D direction a vanishing point is the image of."""
        return _unit(np.linalg.solve(self.K, np.asarray(vp, float)))

    def up(self) -> np.ndarray:
        """The vertical. With no vertical vanishing point the camera is taken as level: the
        image's own vertical. (Its sign does not matter to a plane.)"""
        return self.direction(self.vertical) if self.vertical is not None else np.array([0.0, 1.0, 0.0])

    def wall_normal(self, facade: str) -> np.ndarray | None:
        """The unit normal of a facade's vertical wall plane, or None when this photo has no
        vanishing point for that facade. A photo with no horizontal vanishing point at all is
        a wall facing the camera, and that is what every facade then gets."""
        if not self.horizontal:
            return np.array([0.0, 0.0, 1.0])
        if facade not in self.facades:
            return None
        level = self.direction(self.horizontal[self.facades.index(facade)])
        normal = np.cross(level, self.up())
        return _unit(normal) if np.linalg.norm(normal) > AT_INFINITY else None

    def on_wall(self, pixel, normal: np.ndarray, distance: float) -> np.ndarray | None:
        """The 3-D point where a pixel's ray meets the wall {X : normal·X = distance}."""
        ray = self.ray(pixel)
        facing = float(normal @ ray)
        if abs(facing) < AT_INFINITY:
            return None
        return ray * (distance / facing)

    def project(self, point: np.ndarray) -> np.ndarray:
        """A 3-D point in camera coordinates back to photo pixels."""
        image = self.K @ np.asarray(point, float)
        return image[:2] / image[2]

    def facade_angles(self, a, b) -> list[tuple[float, str]]:
        """Degrees between the segment a→b and the direction from its middle to each facade's
        vanishing point, nearest first. A level line at camera height lies on the horizon, where
        every horizontal vanishing point is: there the first two are nearly equal."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        direction = _unit(b - a)
        middle = (a + b) / 2
        angles = []
        for facade, vp in zip(self.facades, self.horizontal):
            toward = _unit(np.array(vp[:2]) - middle * vp[2]) if vp[2] else np.array(vp[:2])
            angles.append((float(np.degrees(np.arccos(min(1.0, abs(float(direction @ toward)))))), facade))
        return sorted(angles)

    def facade_of(self, a, b, tolerance_deg: float) -> str | None:
        """The facade whose level direction the segment a→b runs toward, within tolerance, or
        None (a vertical feature, or no vanishing point it matches). Never the image slope."""
        angles = self.facade_angles(a, b)
        return angles[0][1] if angles and angles[0][0] <= tolerance_deg else None

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "focal_px": round(self.focal, 2), "focal_35mm": round(self.focal_35mm, 1),
            "focal_source": self.focal_source, "focal_checks": self.focal_checks,
            "principal_point": [round(float(v), 1) for v in self.centre],
            "vertical": [round(v, 4) for v in self.vertical] if self.vertical is not None else None,
            "horizontal": [[round(v, 4) for v in vp] for vp in self.horizontal],
            "facades": list(self.facades),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(width=int(data["width"]), height=int(data["height"]), focal=float(data["focal_px"]),
                   focal_source=data["focal_source"], vertical=data.get("vertical"),
                   horizontal=list(data.get("horizontal") or []), facades=list(data.get("facades") or []),
                   focal_checks=dict(data.get("focal_checks") or {}))


def _off_vertical(vp: list[float], centre: np.ndarray) -> float:
    """Degrees between the direction to a vanishing point from the centre and the photo's vertical."""
    direction = np.array(vp[:2]) if vp[2] == 0.0 else np.array(vp[:2]) - centre
    return float(np.degrees(np.arccos(min(1.0, abs(_unit(direction)[1])))))


def _vertical_ok(vp: list[float] | None, centre: np.ndarray, long_edge: float) -> bool:
    """Can this vanishing point be the vertical: along the photo's vertical, and far enough out
    for a hand-held camera's tilt (VERTICAL_MAX_DEG, VERTICAL_MIN_REACH)."""
    if vp is None or _off_vertical(vp, centre) > VERTICAL_MAX_DEG:
        return False
    return vp[2] == 0.0 or float(np.linalg.norm(np.array(vp[:2]) - centre)) >= VERTICAL_MIN_REACH * long_edge


def _facade_names(horizontal: list, centre: np.ndarray, long_edge: float) -> list[str]:
    """fusion._facades for pixel vanishing points: the one farthest out faces the camera ("front"),
    the others are named by the side of the photo they lie on."""
    reach = [float("inf") if vp[2] == 0.0 else float(np.linalg.norm(np.array(vp[:2]) - centre)) for vp in horizontal]
    names = []
    for vp, far in zip(horizontal, reach):
        x = vp[0] if vp[2] == 0.0 else vp[0] - centre[0]
        front = far == max(reach) if len(horizontal) >= 2 else far >= fusion.FRONT_VP_REACH * long_edge / 2
        names.append("front" if front else ("left" if x < 0 else "right"))
    return names


def _settled_vertical(vertical: list[float] | None, horizontal: list, facades: list[str], centre: np.ndarray,
                      long_edge: float, checks: dict) -> tuple[list[float] | None, list, list[str]]:
    """The vertical, tested for where a hand-held camera can put it. A "vertical" too near the
    centre is a horizontal vanishing point on the horizon; a "facade" point far out along the
    photo's vertical is the vertical. They swap, and the facades are named again. A vertical
    failing with nothing to replace it is dropped (None: refit from uprights, or a level camera).
    Unchanged when the vertical holds."""
    if vertical is None or _vertical_ok(vertical, centre, long_edge):
        return vertical, horizontal, facades
    along = _off_vertical(vertical, centre) <= VERTICAL_MAX_DEG
    candidates = [i for i, vp in enumerate(horizontal) if _vertical_ok(vp, centre, long_edge)]
    if not candidates:
        checks["vertical_rejected"] = [round(v, 1) for v in vertical]
        checks["vertical_why"] = ("too near the centre for a hand-held camera's tilt" if along
                                  else f"more than {VERTICAL_MAX_DEG:.0f} deg off the photo's vertical")
        return None, horizontal, facades

    def reach(vp):
        return float("inf") if vp[2] == 0.0 else float(np.linalg.norm(np.array(vp[:2]) - centre))

    chosen = max(candidates, key=lambda i: reach(horizontal[i]))
    rest = [vp for i, vp in enumerate(horizontal) if i != chosen]
    if along:
        rest.append(list(vertical))     # on the horizon, near the centre: a wall receding from the camera
    checks["vertical_swapped"] = {"was_vertical": [round(v, 1) for v in vertical],
                                  "now_vertical": [round(v, 1) for v in horizontal[chosen]],
                                  "was_facade": facades[chosen] if chosen < len(facades) else None,
                                  "why": "the vertical was too near the centre; a facade point lay far out along "
                                         "the photo's vertical" if along else
                                         f"the vertical was more than {VERTICAL_MAX_DEG:.0f} deg off the photo's vertical"}
    return list(horizontal[chosen]), rest, _facade_names(rest, centre, long_edge)


def _agrees(segments: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Per pixel segment: does it run toward the homogeneous pixel vanishing point?"""
    starts, ends = segments[:, :2], segments[:, 2:]
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    directions = directions / np.maximum(lengths[:, None], 1e-9)
    middles = (starts + ends) / 2
    if abs(vp[2]) < AT_INFINITY:
        toward = np.tile(_unit(vp[:2]), (len(segments), 1))
    else:
        toward = vp[:2] / vp[2] - middles
        toward = toward / np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
    return np.abs(np.einsum("ij,ij->i", toward, directions)) >= np.cos(np.radians(AGREE_DEG))


def _vertical_from_segments(segments: np.ndarray, size: tuple[int, int]) -> list[float] | None:
    """The vertical vanishing point fitted to the photo's near-vertical segments (wall corners,
    trim, window sides), weighted by length: [x, y, 1] px, or a direction at infinity."""
    if segments is None or not len(segments):
        return None
    directions = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(directions, axis=1)
    upright = np.abs(directions[:, 1]) / np.maximum(lengths, 1e-9) >= np.cos(np.radians(UPRIGHT_DEG))
    if upright.sum() < UPRIGHT_MIN_SEGMENTS:
        return None
    return _fit_vp(segments[upright], size)


def _fit_vp(chosen: np.ndarray, size: tuple[int, int]) -> list[float]:
    """The point the segments run toward, least squares weighted by length: [x, y, 1] px, or a
    direction [dx, dy, 0] at infinity."""
    width, height = size
    centre, s = np.array([width / 2.0, height / 2.0]), max(width, height) / 2.0
    lengths = np.linalg.norm(chosen[:, 2:] - chosen[:, :2], axis=1)
    p1 = np.hstack([(chosen[:, :2] - centre) / s, np.ones((len(chosen), 1))])
    p2 = np.hstack([(chosen[:, 2:] - centre) / s, np.ones((len(chosen), 1))])
    lines = np.cross(p1, p2)
    lines = lines / np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
    vp = np.linalg.svd(lines * np.sqrt(lengths)[:, None], full_matrices=False)[2][-1]
    if abs(vp[2]) < 1e-9:
        return [float(v) for v in _unit(vp[:2])] + [0.0]
    x, y = vp[:2] / vp[2] * s + centre
    return [float(x), float(y), 1.0]


def _second_facade(segments: np.ndarray, first: list[float], vertical: list[float] | None,
                   size: tuple[int, int]) -> tuple[list[float], float] | None:
    """The perpendicular wall's vanishing point detection did not find, and the focal length it
    implies. For a focal length f, the first facade's level direction d and the vertical u give
    the other facade's direction u × d, so its vanishing point K·(u × d) is fixed; the f whose
    point the photo's own segments run toward is the one kept. None when no f is supported."""
    if segments is None or len(segments) < SECOND_MIN_SEGMENTS:
        return None
    width, height = size
    long_edge = float(max(width, height))
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    explained = _agrees(segments, np.asarray(first, float))
    best = None
    for f35 in FOCAL_SWEEP_35MM:
        cam = Camera(width, height, _from_35mm(f35, long_edge), "vp", vertical, [first], [])
        up = cam.up()
        level = cam.direction(first)
        level = _unit(level - (level @ up) * up)
        other = np.cross(up, level)
        vp = cam.K @ other
        vp = vp / vp[2] if abs(vp[2]) > AT_INFINITY else np.array([*_unit(vp[:2]), 0.0])
        upright = _agrees(segments, cam.K @ up)
        free = ~explained & ~upright
        support = float(lengths[_agrees(segments, vp) & free].sum())
        if best is None or support > best[0]:
            best = (support, vp, cam.focal, _agrees(segments, vp) & free, float(lengths[free].sum()))
    support, vp, focal, members, pool = best
    if members.sum() < SECOND_MIN_SEGMENTS or pool <= 0 or support < SECOND_SHARE * pool:
        return None
    # The sweep's point is only as good as its focal step and the vertical; the segments that
    # run toward it say exactly where it is. The focal length then follows from the refit point.
    refined = _fit_vp(segments[members], size)
    centre = np.array([width / 2.0, height / 2.0])
    refit = _focal_from_pair(first, refined, centre)
    return refined, refit if refit is not None else focal


def build(scene: partition.Scene | None, size: tuple[int, int], focal_35mm: float | None) -> Camera:
    """The camera for a photo from its vanishing points and, when the upload had one, its EXIF
    focal length. See the module docstring for which focal length wins."""
    width, height = size
    long_edge = float(max(width, height))
    centre = np.array([width / 2.0, height / 2.0])
    vertical = horizontal = None
    facades: list[str] = []
    checks: dict = {}
    segments = None
    if scene is not None:
        vertical = _pixel_vp(scene.vertical, scene) if scene.vertical is not None else None
        horizontal = [_pixel_vp(vp, scene) for vp in scene.horizontal]
        facades = fusion._facades(scene)
        segments = np.asarray(scene.segments, float)
        on_house = getattr(scene, "on_house", None)
        if on_house is not None and len(on_house) == len(segments) and np.any(on_house):
            segments = segments[np.asarray(on_house, bool)]
            checks["segments"] = "target house only"
    # Not the vertical (off the photo's vertical, or too near the centre): a facade point far out
    # along the vertical takes its place, else it is fitted afresh from the photo's upright
    # segments, else the camera is level.
    vertical, horizontal, facades = _settled_vertical(vertical, horizontal or [], facades, centre, long_edge, checks)
    if vertical is None and segments is not None:
        fitted = _vertical_from_segments(segments, size)
        if _vertical_ok(fitted, centre, long_edge):
            vertical = fitted
            checks["vertical"] = "fitted to upright segments"
        elif fitted is not None:
            checks["vertical_fit_rejected"] = [round(v, 1) for v in fitted]
    found_second = None
    if segments is not None and horizontal and len(horizontal) == 1:
        found_second = _second_facade(segments, horizontal[0], vertical, size)
        if found_second is not None and focal_spread(horizontal[0], found_second[0], centre) > FOCAL_SPREAD_MAX:
            # The first facade faces the camera: where the sweep put the other one, and the focal
            # length it implies, are as uncertain as the pair is ill-conditioned.
            checks["second_facade_rejected"] = "ill-conditioned with the first facade"
            found_second = None
        if found_second is not None:
            vp, _ = found_second
            horizontal = horizontal + [vp]
            # Detection's name for its own facade stands (lines already carry it as their side);
            # the found one is named by which side of the photo its vanishing point lies on.
            x = vp[0] - centre[0] if vp[2] else vp[0]
            name = "left" if x < 0 else "right"
            facades = facades + [name if name != facades[0] else "front"]
            checks["second_facade"] = "found by its focal sweep"

    estimates, rejected = _pair_estimates(horizontal or [], vertical, centre, long_edge)
    if rejected:
        checks["vp_rejected"] = rejected

    if found_second is not None:
        estimates.append(found_second[1])
    focal, source = _choose_focal(estimates, focal_35mm, long_edge, checks)
    found = Camera(width=width, height=height, focal=focal, focal_source=source, vertical=vertical,
                   horizontal=horizontal or [], facades=facades, focal_checks=checks)
    if len(found.horizontal) == 1 and source in ("vp", "exif"):
        # One facade seen, and the photo gave the focal length: a house's walls meet square, so
        # the other facade's vanishing point follows from the first and the vertical. Only with
        # a measured focal length (the perspective's or the lens's own); an assumed one would
        # place it wherever the guess put it (the scale solve tries that one only when the
        # gables' rakes confirm it: services/scale.py).
        _add_square_facade(found)
    return found


SQUARE_NOTE = "square to the first (assumed square corners)"


def _pair_estimates(horizontal: list, vertical: list[float] | None, centre: np.ndarray,
                    long_edge: float) -> tuple[list[float], list[dict]]:
    """f from every well-conditioned perpendicular pair, and the pairs dropped as noise."""
    estimates, rejected = [], []
    for i, a in enumerate(horizontal):
        for b in horizontal[i + 1:] + ([vertical] if vertical is not None else []):
            found = _focal_from_pair(a, b, centre)
            if found is None or not PLAUSIBLE_FOCAL_35MM[0] <= _to_35mm(found, long_edge) <= PLAUSIBLE_FOCAL_35MM[1]:
                continue
            spread = focal_spread(a, b, centre)
            if spread > FOCAL_SPREAD_MAX:
                rejected.append({"focal_35mm": round(_to_35mm(found, long_edge), 1),
                                 "spread": round(spread, 3) if np.isfinite(spread) else None,
                                 "with": "vertical" if b is vertical else "facade"})
                continue
            estimates.append(found)
    return estimates, rejected


def _choose_focal(estimates: list[float], focal_35mm: float | None, long_edge: float,
                  checks: dict) -> tuple[float, str]:
    """The vanishing points' focal, else the lens's own (EXIF), else a typical phone's."""
    focal = source = None
    if estimates:
        focal, source = float(np.median(estimates)), "vp"
        checks["vp"] = round(_to_35mm(focal, long_edge), 1)
    if focal_35mm:
        checks["exif"] = round(float(focal_35mm), 1)
        if not estimates:
            focal, source = _from_35mm(float(focal_35mm), long_edge), "exif"
        else:
            checks["agree"] = abs(checks["vp"] - focal_35mm) <= EXIF_AGREES * focal_35mm
    if focal is None:
        focal, source = _from_35mm(ASSUMED_FOCAL_35MM, long_edge), "assumed"
    return focal, source


def _add_square_facade(found: Camera) -> None:
    """The second facade's vanishing point, square to the only one seen, under found's focal."""
    up = found.up()
    level = found.direction(found.horizontal[0])
    other = found.K @ np.cross(up, _unit(level - (level @ up) * up))
    vp = [float(v) for v in (other / other[2])] if abs(other[2]) > AT_INFINITY else [*map(float, _unit(other[:2])), 0.0]
    x = vp[0] - found.centre[0] if vp[2] else vp[0]
    name = "left" if x < 0 else "right"
    found.horizontal.append(vp)
    found.facades.append(name if name != found.facades[0] else "front")
    found.focal_checks["second_facade"] = SQUARE_NOTE


def observed(cam: Camera) -> tuple[list, list[str]]:
    """The camera's facades read from the photo, without one made square to them."""
    if cam.focal_checks.get("second_facade") == SQUARE_NOTE and len(cam.horizontal) > 1:
        return [list(vp) for vp in cam.horizontal[:-1]], list(cam.facades[:-1])
    return [list(vp) for vp in cam.horizontal], list(cam.facades)


def with_focal(cam: Camera, focal_35mm: float, source: str, square: bool = True) -> Camera:
    """The same vanishing points under another focal length; a facade made square to the only
    one seen is made again under it (when `square`)."""
    horizontal, facades = observed(cam)
    checks = {k: v for k, v in cam.focal_checks.items() if k != "second_facade"}
    found = Camera(width=cam.width, height=cam.height, focal=_from_35mm(focal_35mm, max(cam.width, cam.height)),
                   focal_source=source, vertical=cam.vertical, horizontal=horizontal, facades=facades,
                   focal_checks=checks)
    if square and len(found.horizontal) == 1:
        _add_square_facade(found)
    return found


def perturbed(cam: Camera, which: str, degrees: float = 0.0, focal_scale: float = 1.0) -> Camera | None:
    """The same camera with one vanishing point turned by `degrees` about the centre (`which`: a
    facade's name or "vertical"), or with its focal scaled (`which`="focal"): what a solve is
    re-run under to see how far each length moves. A facade made square to the only one seen is
    made again, so it follows. None when there is no such vanishing point."""
    horizontal, facades = observed(cam)
    vertical = list(cam.vertical) if cam.vertical is not None else None

    def turn(vp: list[float]) -> list[float]:
        if vp[2] == 0.0:
            angle = np.radians(degrees)
            x, y = float(vp[0]), float(vp[1])
            return [float(np.cos(angle) * x - np.sin(angle) * y), float(np.sin(angle) * x + np.cos(angle) * y), 0.0]
        return _turned(vp, cam.centre, degrees)

    if which == "vertical":
        if vertical is None:
            return None
        vertical = turn(vertical)
    elif which in facades:
        index = facades.index(which)
        horizontal[index] = turn(horizontal[index])
    elif which != "focal":
        return None
    checks = {k: v for k, v in cam.focal_checks.items() if k != "second_facade"}
    found = Camera(width=cam.width, height=cam.height, focal=cam.focal * focal_scale, focal_source=cam.focal_source,
                   vertical=vertical, horizontal=horizontal, facades=facades, focal_checks=checks)
    if cam.focal_checks.get("second_facade") == SQUARE_NOTE and len(found.horizontal) == 1:
        _add_square_facade(found)
        # A small turn keeps the made facade's name; should it flip sides, it is still the same wall.
        found.facades = list(cam.facades)[:len(found.facades)]
    return found


def rechecked(cam: Camera) -> Camera:
    """A stored camera under the current focal rules: its pairs of vanishing points tested again
    for conditioning (a camera stored before a rule changed keeps whatever focal it had then).
    The same camera when nothing changes. Its vertical is tested again too (_settled_vertical)."""
    horizontal, facades = observed(cam)
    long_edge = float(max(cam.width, cam.height))
    checks = {k: v for k, v in cam.focal_checks.items() if k not in ("vp", "vp_rejected", "agree", "second_facade")}
    vertical, horizontal, facades = _settled_vertical(cam.vertical, horizontal, facades, cam.centre, long_edge, checks)
    settled = vertical is not cam.vertical
    if not settled and cam.focal_source != "vp":
        return cam
    estimates, rejected = _pair_estimates(horizontal, vertical, cam.centre, long_edge)
    if rejected:
        checks["vp_rejected"] = rejected
    focal, source = _choose_focal(estimates, checks.get("exif"), long_edge, checks)
    # Stored vanishing points are rounded: the same pairs give the same focal to within that.
    if not settled and source == "vp" and abs(focal - cam.focal) <= 0.01 * cam.focal:
        return cam
    if not settled:
        checks["rechecked"] = f"stored focal {round(cam.focal_35mm, 1)} mm failed today's conditioning test"
    found = Camera(width=cam.width, height=cam.height, focal=focal, focal_source=source, vertical=vertical,
                   horizontal=horizontal, facades=facades, focal_checks=checks)
    if len(found.horizontal) == 1 and source in ("vp", "exif"):
        _add_square_facade(found)
    return found
