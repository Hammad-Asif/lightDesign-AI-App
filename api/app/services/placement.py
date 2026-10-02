"""Bulbs along lit lines, at real spacing, sized and placed by the camera (M3 Day 3).

A line's 3-D position comes from services/scale.py. Bulbs are laid every `spacing` inches along
it in 3-D and projected through the camera, which is the cross-ratio result: even real spacing
becomes the right uneven spacing in the photo. A bulb's drawn size is f · diameter / depth, so a
C9 near the camera draws larger than one at the far end.

Continuous runs (Lukas): lit level lines of the same kind (eave to eave, ridge to ridge) whose
ends meet at the same roof level, lit with the same settings, are one path, and the spacing
carries through the corner without resetting. The path ends where the roof steps (the ends'
heights differ) or where the kind changes, e.g. an eave meeting a rake.

A line the geometry could not place in 3-D but the salesperson overrode (its real middle dragged
onto it, its length typed) is walked with the client's midpoint mapping (scale.override_fraction).

Bulbs behind an object (Region.hidden) are counted and flagged hidden, never drawn: the lights are
installed there all the same.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from app.schemas import LightingSettings, Pattern, PatternStep
from app.services import patterns, products, scale
from app.services.camera import Camera

LINE_CATEGORIES = {"c9", "c7", "mini", "icicle", "garland"}
CONTINUOUS_EDGES = {"eave", "ridge", "junction"}
SAME_LEVEL_IN = 6.0            # two runs' meeting ends this close in height are one roof level
GARLAND_SPACING_IN = 4.0       # Lukas: lights every 4 in within the band, when the product has none
GARLAND_BAND_IN = 10.0         # Lukas: a band about 10 in wide, when the product has no thickness
GARLAND_ACROSS = 0.3           # bulbs sit this share of the band's width either side of its middle
ICICLE_BULBS_PER_DROP = 4      # when the product does not say (bulb_count per strand / drops per strand)
DEFAULT_COLOUR = "warm_white"  # Lukas: the default white is warm white


@dataclass
class Member:
    region_id: str
    pixels: list[list[float]]            # the drawn polyline, in the path's direction
    points3d: list[np.ndarray] | None    # the same ends in 3-D, or None (override)
    hidden: list[list[float]]            # [start, end] px along the drawn polyline, in its stored direction
    reversed: bool = False
    override: dict | None = None
    length_in: float | None = None


@dataclass
class Path:
    members: list[Member]
    settings: dict
    product: object
    edge: str | None
    bulbs: dict = field(default_factory=dict)     # region id -> list of bulbs
    drops: dict = field(default_factory=dict)     # region id -> list of [x0, y0, x1, y1]
    band: dict = field(default_factory=dict)      # region id -> list of [x, y, width px]


def _hex(colour: str) -> str:
    return products.COLOURS.get(colour, products.COLOURS[DEFAULT_COLOUR])[1]


def _pattern(settings: LightingSettings, product, kind: str) -> tuple[Pattern, float | None]:
    pattern, spacing, _, _ = patterns.resolve(settings, product, kind)
    return pattern or Pattern(steps=[PatternStep(colour=DEFAULT_COLOUR)]), spacing


def _along(pixels: list[list[float]], point: np.ndarray) -> float:
    """Distance in px along a polyline to the point on it nearest `point`."""
    best, walked, best_at = None, 0.0, 0.0
    for a, b in zip(pixels[:-1], pixels[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        span = float(np.linalg.norm(b - a))
        t = 0.0 if span < 1e-9 else float(np.clip((point - a) @ (b - a) / (span * span), 0, 1))
        gap = float(np.linalg.norm(point - (a + t * (b - a))))
        if best is None or gap < best:
            best, best_at = gap, walked + t * span
        walked += span
    return best_at


def _is_hidden(member: Member, pixel: np.ndarray) -> bool:
    if not member.hidden:
        return False
    stored = member.pixels[::-1] if member.reversed else member.pixels
    at = _along(stored, pixel)
    return any(start <= at <= end for start, end in member.hidden)


def _chain(members: list[Member], edges: dict, camera: Camera, up: np.ndarray, reach: float) -> list[list[Member]]:
    """Members joined into continuous paths: level lines of one kind meeting at one roof level."""
    left = list(members)
    paths = []

    def meets(end_pixel, end_point, other: Member) -> tuple[bool, bool] | None:
        """Does `other` start (False) or end (True) where this path ends?"""
        if other.points3d is None:
            return None
        for flip, (pixel, point) in ((False, (other.pixels[0], other.points3d[0])),
                                     (True, (other.pixels[-1], other.points3d[-1]))):
            close = np.hypot(pixel[0] - end_pixel[0], pixel[1] - end_pixel[1]) <= reach
            level = abs(float((point - end_point) @ up)) <= SAME_LEVEL_IN
            if close and level:
                return True, flip
        return None

    while left:
        path = [left.pop(0)]
        if path[0].points3d is not None and edges.get(path[0].region_id) in CONTINUOUS_EDGES:
            for forward in (True, False):
                while True:
                    tip = path[-1] if forward else path[0]
                    end_pixel = tip.pixels[-1] if forward else tip.pixels[0]
                    end_point = tip.points3d[-1] if forward else tip.points3d[0]
                    found = None
                    for other in left:
                        if edges.get(other.region_id) != edges.get(tip.region_id):
                            continue
                        hit = meets(end_pixel, end_point, other)
                        if hit:
                            found = (other, hit[1])
                            break
                    if not found:
                        break
                    other, at_end = found
                    left.remove(other)
                    # Walking forward the next member must start at the meeting point; backward, end there.
                    if at_end == forward:
                        other.pixels, other.points3d = other.pixels[::-1], other.points3d[::-1]
                        other.reversed = not other.reversed
                    if forward:
                        path.append(other)
                    else:
                        path.insert(0, other)
        paths.append(path)
    return paths


def _walk_3d(path: list[Member], spacing: float):
    """(member, 3-D point, direction) every `spacing` inches along the path, from its start."""
    segments = []
    for member in path:
        points = member.points3d
        for a, b in zip(points[:-1], points[1:]):
            segments.append((member, a, b))
        # The step from one member's end to the next member's start (the corner) is walked too.
    joined = []
    for index, (member, a, b) in enumerate(segments):
        joined.append((member, a, b))
        if index + 1 < len(segments) and segments[index + 1][0] is not member:
            joined.append((member, b, segments[index + 1][1]))
    total = sum(float(np.linalg.norm(b - a)) for _, a, b in joined)
    count = int(np.floor(total / spacing + 1e-6)) + 1 if total > 0 else 0
    out, index, walked = [], 0, 0.0
    for k in range(count):
        target = k * spacing
        while index < len(joined) - 1 and walked + float(np.linalg.norm(joined[index][2] - joined[index][1])) < target - 1e-9:
            walked += float(np.linalg.norm(joined[index][2] - joined[index][1]))
            index += 1
        member, a, b = joined[index]
        span = float(np.linalg.norm(b - a))
        t = 0.0 if span < 1e-9 else min(1.0, (target - walked) / span)
        out.append((member, a + t * (b - a), (b - a) / span if span > 1e-9 else np.zeros(3)))
    return out, total


def _walk_override(member: Member, spacing: float):
    """(pixel, px per inch) every `spacing` inches along an overridden line, by the midpoint mapping."""
    a, b = np.asarray(member.pixels[0], float), np.asarray(member.pixels[-1], float)
    drawn = float(np.linalg.norm(b - a))
    length = float(member.length_in)
    mid = member.override.get("mid")
    u_mid = 0.5 if not mid else float(np.clip((np.asarray(mid, float) - a) @ (b - a) / max(drawn * drawn, 1e-9), 0.05, 0.95))
    if member.reversed:
        u_mid = 1 - u_mid
    count = int(np.floor(length / spacing + 1e-6)) + 1
    out = []
    for k in range(count):
        t = min(1.0, k * spacing / length)
        u = scale.override_fraction(t, u_mid)
        h = 1e-4
        slope = (scale.override_fraction(min(1.0, t + h), u_mid) - scale.override_fraction(max(0.0, t - h), u_mid)) / (
            min(1.0, t + h) - max(0.0, t - h))
        out.append((a + u * (b - a), drawn * slope / length))
    return out, length


def place_path(path: Path, camera: Camera, up: np.ndarray) -> float:
    """Fills path.bulbs (and drops / band); returns the path's real length in inches."""
    settings = LightingSettings(**path.settings)
    category = path.product.category
    pattern, spacing = _pattern(settings, path.product, "roof")
    if category == "garland":
        spacing = path.product.spacing_in or GARLAND_SPACING_IN
    diameter = float(path.product.bulb_diameter_in or 0.2)
    band_in = float(path.product.diameter_in or GARLAND_BAND_IN)
    drops_in = [float(d) for d in (path.product.drops_in or [])] or [12.0]
    per_drop = ICICLE_BULBS_PER_DROP
    if category == "icicle" and path.product.bulb_count and path.product.strand_length_ft and spacing:
        per_drop = max(1, round(path.product.bulb_count / (path.product.strand_length_ft * 12 / spacing)))

    stations = []   # (member, pixel, px per inch, 3-D point or None, direction or None)
    if path.members[0].points3d is not None:
        walked, total = _walk_3d(path.members, spacing)
        for member, point, direction in walked:
            depth = float(point[2])
            stations.append((member, camera.project(point), camera.focal / depth, point, direction))
    else:
        member = path.members[0]
        walked, total = _walk_override(member, spacing)
        stations = [(member, pixel, per_inch, None, None) for pixel, per_inch in walked]

    colours = patterns.colours(pattern, len(stations))
    for index, ((member, pixel, per_inch, point, direction), colour) in enumerate(zip(stations, colours)):
        hidden = _is_hidden(member, pixel)
        bulbs = path.bulbs.setdefault(member.region_id, [])
        if category == "icicle":
            length = drops_in[index % len(drops_in)]
            if point is not None:
                bottom = camera.project(point - up * length)
            else:
                bottom = pixel + np.array([0.0, per_inch * length])
            path.drops.setdefault(member.region_id, []).append([*map(float, pixel), *map(float, bottom)])
            for j in range(1, per_drop + 1):
                share = j / per_drop
                at = camera.project(point - up * length * share) if point is not None else pixel + np.array([0.0, per_inch * length * share])
                size = camera.focal * diameter / float((point - up * length * share)[2]) if point is not None else per_inch * diameter
                bulbs.append([float(at[0]), float(at[1]), float(size), _hex(colour), hidden])
            continue
        if category == "garland":
            across_dir = up - (up @ direction) * direction if direction is not None else None
            side = GARLAND_ACROSS * band_in * (1 if index % 2 == 0 else -1)
            if point is not None and np.linalg.norm(across_dir) > 1e-9:
                across_dir = across_dir / np.linalg.norm(across_dir)
                at = camera.project(point + across_dir * side)
            else:
                at = pixel + np.array([0.0, -per_inch * side])
            path.band.setdefault(member.region_id, []).append([float(pixel[0]), float(pixel[1]), float(per_inch * band_in)])
            bulbs.append([float(at[0]), float(at[1]), float(per_inch * diameter), _hex(colour), hidden])
            continue
        bulbs.append([float(pixel[0]), float(pixel[1]), float(per_inch * diameter), _hex(colour), hidden])
    return total


def place(camera: Camera, regions: list, settings_by_region: dict, products_by_id: dict,
          solution: scale.Solution, overrides: dict | None = None) -> dict:
    """Bulbs for every lit line with a line product, by region id:
    {"bulbs": [[x, y, diameter_px, colour hex, hidden]], "count", "hidden", "length_in",
     "path": index of the continuous run it belongs to, "drops", "band"} or {"needs": reason}."""
    overrides = overrides or {}
    up = scale.world_up(camera)
    reach = scale.CORNER_REACH_FRACTION * max(camera.width, camera.height)
    out: dict = {}
    groups: dict[str, list[Member]] = {}
    edges: dict[str, str | None] = {}
    product_of: dict[str, object] = {}
    for region in regions:
        region_id = str(region.id)
        settings = settings_by_region.get(region_id)
        if not settings or region.geometry != "line" or not region.polyline or len(region.polyline) < 2:
            continue
        product = products_by_id.get(str(settings.get("product_id")))
        if product is None or product.category not in LINE_CATEGORIES:
            continue
        geometry = solution.lines.get(region_id)
        override = overrides.get(region_id)
        pixels = [list(map(float, p)) for p in region.polyline]
        if geometry is not None and geometry.points3d:
            member = Member(region_id, pixels, list(geometry.points3d), list(region.hidden or []))
        elif override and override.get("length_in"):
            member = Member(region_id, pixels, None, list(region.hidden or []), override=override,
                            length_in=float(override["length_in"]))
        else:
            out[region_id] = {"needs": (geometry.needs if geometry else None) or "measurement"}
            continue
        key = json.dumps(settings, sort_keys=True)
        groups.setdefault(key, []).append(member)
        edges[region_id] = region.edge
        product_of[key] = product

    number = 0
    for key, members in groups.items():
        for chained in _chain(members, edges, camera, up, reach):
            path = Path(chained, json.loads(key), product_of[key], edges.get(chained[0].region_id))
            total = place_path(path, camera, up)
            for member in chained:
                bulbs = path.bulbs.get(member.region_id, [])
                out[member.region_id] = {
                    "bulbs": [[round(b[0], 1), round(b[1], 1), round(b[2], 2), b[3], b[4]] for b in bulbs],
                    "count": len(bulbs), "hidden": sum(1 for b in bulbs if b[4]),
                    "path": number, "path_length_in": round(total, 1),
                    "drops": [[round(v, 1) for v in d] for d in path.drops.get(member.region_id, [])],
                    "band": [[round(v, 1) for v in b] for b in path.band.get(member.region_id, [])],
                }
            number += 1
    return out


# -- areas: palm trunks, canopies and fronds, wreaths (M3 Day 4) ----------------------------------

SCATTER_KINDS = {"hedge", "shrub", "topiary", "tree_canopy", "palm_frond"}
DENSITY = {"light": 0.5, "medium": 1.0, "heavy": 2.0}    # Lukas: of the product's per-10-sq-ft figure
DEFAULT_DENSITY_PER_10SQFT = 100.0                        # Lukas's medium, for a product without one
DRAW_CAP = 4000                    # at most this many scattered bulbs are drawn; all are counted
TRUNK_WIDTH_ROWS = (0.2, 0.8)      # the trunk's width is its median over this middle band of height
WREATH_RING = 0.42                 # bulbs sit on a ring this share of the wreath's diameter out
DEFAULT_WREATH_IN = 36             # Lukas: default wreath size


def wall_scale(camera: Camera, solution: scale.Solution, lines: dict, pixel) -> tuple[float, str] | None:
    """(px per inch, facade) at a pixel, read on the nearest wall: the wall whose placed lines sit
    above that spot in the photo, horizontally; the nearest such line when none spans it. None
    until a measurement has made a wall metric. (The planes run on forever, so the nearest plane
    along the ray would be a side wall's extension passing in front of the house.)"""
    x = float(pixel[0])
    best = None
    for line_id, placed in solution.lines.items():
        facade = placed.wall
        if facade is None or facade not in solution.walls or line_id not in lines:
            continue
        xs = [p[0] for p in lines[line_id]]
        gap = max(0.0, min(xs) - x, x - max(xs))
        if best is None or gap < best[0]:
            best = (gap, facade)
    if best is None:
        return None
    facade = best[1]
    point = camera.on_wall(pixel, camera.wall_normal(facade), solution.walls[facade]["distance_in"])
    if point is None or point[2] <= 0:
        return None
    return camera.focal / float(point[2]), facade


def _base(mask: np.ndarray) -> list[float]:
    """The bottom-centre of a mask: where it stands, and so the depth it is read at."""
    ys, xs = np.nonzero(mask)
    bottom = ys.max()
    low = ys >= bottom - max(2, (bottom - ys.min()) * 0.05)
    return [float(xs[low].mean()), float(bottom)]


def _colours_for(settings: LightingSettings, product, kind: str, count: int, turns: list[int] | None = None) -> list[str]:
    pattern, *_ = patterns.resolve(settings, product, kind)
    pattern = pattern or Pattern(steps=[PatternStep(colour=DEFAULT_COLOUR)])
    if turns is not None and pattern.unit == "turn":
        per_turn = patterns.colours(pattern, max(turns) + 1 if turns else 0)
        return [_hex(per_turn[t]) for t in turns]
    return [_hex(c) for c in patterns.colours(pattern, count)]


def spiral(mask: np.ndarray, settings: LightingSettings, product, kind: str, per_inch: float,
           override: dict) -> dict:
    """A strand wound up a palm trunk. The trunk's real diameter and height are its mask's pixel
    size at the local scale; the wrap runs between the settings' start and stop at their turn
    spacing. Bulbs on the far half of a turn are counted and flagged hidden, never drawn. The
    strand length is an estimate the salesperson can replace (override `strand_ft`)."""
    _, _, _, wrap = patterns.resolve(settings, product, kind)
    rows = np.flatnonzero(mask.any(axis=1))
    top, bottom = int(rows.min()), int(rows.max())
    height_px = max(1, bottom - top)
    band = rows[(rows >= top + TRUNK_WIDTH_ROWS[0] * height_px) & (rows <= top + TRUNK_WIDTH_ROWS[1] * height_px)]
    widths = [np.ptp(np.flatnonzero(mask[r])) + 1 for r in (band if band.size else rows)]
    diameter_in = float(np.median(widths)) / per_inch
    height_in = height_px / per_inch
    zone_low, zone_high = bottom - wrap.start * height_px, bottom - wrap.stop * height_px
    zone_in = (zone_low - zone_high) / per_inch
    turns = zone_in / wrap.turn_spacing_in
    estimate_in = turns * float(np.hypot(np.pi * diameter_in, wrap.turn_spacing_in))
    strand_in = float(override["strand_ft"]) * 12 if override.get("strand_ft") else estimate_in
    spacing = float(product.spacing_in or 4.0)
    count = int(np.floor(strand_in / spacing + 1e-6)) + 1 if strand_in > 0 else 0

    bulbs, turn_of = [], []
    for k in range(count):
        share = min(1.0, k * spacing / strand_in) if strand_in > 0 else 0.0
        angle = 2 * np.pi * turns * share
        y = zone_low - share * (zone_low - zone_high)
        row = int(np.clip(round(y), top, bottom))
        xs = np.flatnonzero(mask[row])
        if not xs.size:
            near = rows[np.argmin(np.abs(rows - row))]
            xs = np.flatnonzero(mask[near])
        centre, half = (xs[0] + xs[-1]) / 2, (xs[-1] - xs[0] + 1) / 2
        # Angle 0 is the trunk's right edge; the near half faces the camera (sin > 0).
        bulbs.append([float(centre + half * np.cos(angle)), float(y), float(product.bulb_diameter_in or 0.2) * per_inch,
                      None, bool(np.sin(angle) < 0)])
        turn_of.append(int(angle // (2 * np.pi)))
    for bulb, colour in zip(bulbs, _colours_for(settings, product, kind, len(bulbs), turn_of)):
        bulb[3] = colour
    # Estimated until the salesperson types the strand length; the count then follows from it.
    return {"bulbs": bulbs, "count": len(bulbs), "hidden": sum(1 for b in bulbs if b[4]),
            "estimated": not override.get("strand_ft"),
            "strand_ft": round(strand_in / 12, 1), "strand_estimate_ft": round(estimate_in / 12, 1),
            "strand_set": bool(override.get("strand_ft")),
            "trunk": {"diameter_in": round(diameter_in, 1), "height_ft": round(height_in / 12, 1),
                      "turns": round(turns, 1)}}


def scatter(mask: np.ndarray, settings: LightingSettings, product, kind: str, per_inch: float, override: dict,
            seed: str) -> dict:
    """Mini lights over a canopy, fronds or a hedge. The area in a photo is a projected shape, not
    a surface (Lukas), so the count is an estimate: square feet at the local scale × the product's
    density × the preset; the salesperson's count replaces it. That many bulbs are scattered over
    the mask, the same way every time."""
    area_sqft = float(mask.sum()) / (per_inch * per_inch) / 144.0
    preset = override.get("density") or "medium"
    density = float(getattr(product, "density_per_10sqft", None) or DEFAULT_DENSITY_PER_10SQFT)
    estimate = int(round(area_sqft / 10.0 * density * DENSITY[preset]))
    count = int(override["count"]) if override.get("count") is not None else estimate
    ys, xs = np.nonzero(mask)
    # Seeded by the region, so the same bulbs are drawn every time the photo is opened.
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    drawn = min(count, DRAW_CAP)
    picks = rng.choice(len(xs), size=drawn, replace=drawn > len(xs)) if len(xs) and drawn else []
    size = float(product.bulb_diameter_in or 0.2) * per_inch
    colours = _colours_for(settings, product, kind, drawn)
    bulbs = [[float(xs[i]), float(ys[i]), size, colour, False] for i, colour in zip(picks, colours)]
    return {"bulbs": bulbs, "count": count, "hidden": 0, "estimated": override.get("count") is None,
            "estimate": estimate, "area_sqft": round(area_sqft, 1), "density": preset, "drawn": drawn}


def wreath(centre, product, size_in: float, per_inch: float) -> dict:
    """A wreath where the salesperson tapped: a ring of its bulbs, at the local scale."""
    count = int(getattr(product, "bulb_count", None) or 100)
    diameter_px = size_in * per_inch
    colours = [_hex(c) for c in (getattr(product, "colours", None) or [DEFAULT_COLOUR])]
    bulbs = []
    for k in range(count):
        angle = 2 * np.pi * k / count
        bulbs.append([float(centre[0] + WREATH_RING * diameter_px * np.cos(angle)),
                      float(centre[1] + WREATH_RING * diameter_px * np.sin(angle)),
                      0.2 * per_inch, colours[k % len(colours)], False])
    return {"bulbs": bulbs, "count": count, "hidden": 0, "wreaths": 1, "size_in": size_in,
            "ring": [float(centre[0]), float(centre[1]), float(diameter_px)]}


def place_areas(camera: Camera, regions: list, settings_by_region: dict, products_by_id: dict,
                solution: scale.Solution, overrides: dict, masks: dict) -> dict:
    """Bulbs for lit palm trunks and collars (spirals), canopies, fronds and hedges (estimates), and
    every wreath (tapped, lit or not: Lukas's default is a 36 in warm-white one), by region id."""
    lines = {str(r.id): r.polyline for r in regions if r.geometry == "line" and r.polyline}
    wreaths = {int(p.diameter_in): p for p in products_by_id.values()
               if p.category == "wreath" and p.owner_id is None and p.diameter_in and not getattr(p, "archived", False)}
    out: dict = {}
    for region in regions:
        region_id = str(region.id)
        mask = masks.get(region_id)
        if region.geometry != "area" or mask is None or not mask.any():
            continue
        override = overrides.get(region_id) or {}
        settings = settings_by_region.get(region_id)
        product = products_by_id.get(str(settings.get("product_id"))) if settings else None
        if region.kind == "wreath":
            size = float(override.get("size_in") or (product.diameter_in if product and product.category == "wreath" else 0)
                         or DEFAULT_WREATH_IN)
            chosen = product if product is not None and product.category == "wreath" else wreaths.get(int(size))
            centre = override.get("at") or [float(v) for v in np.argwhere(mask).mean(axis=0)[::-1]]
            local = wall_scale(camera, solution, lines, centre)
            out[region_id] = wreath(centre, chosen, size, local[0]) if local else {"needs": "measurement"}
            continue
        if not settings or product is None or product.category in ("wreath", "icicle", "garland"):
            continue
        if region.kind not in patterns.WRAP_KINDS and region.kind not in SCATTER_KINDS:
            continue
        local = wall_scale(camera, solution, lines, _base(mask))
        if local is None:
            out[region_id] = {"needs": "measurement"}
            continue
        parsed = LightingSettings(**settings)
        if region.kind in patterns.WRAP_KINDS:
            out[region_id] = spiral(mask, parsed, product, region.kind, local[0], override)
        else:
            out[region_id] = scatter(mask, parsed, product, region.kind, local[0], override, region_id)
        out[region_id]["scale_from"] = f"the {local[1]} wall"
    for placed in out.values():
        for bulb in placed.get("bulbs", []):
            bulb[:3] = [round(bulb[0], 1), round(bulb[1], 1), round(bulb[2], 2)]
    return out
