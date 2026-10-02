"""One roof-line detection path. V3 draws the roof lines; SAM3 detects and segments the rest.

    SAM3: roof planes + blockers + every other requested kind
      -> target house (services/target.py) -> padded crop
      -> V3 on the crop (cached) -> normalize_v3_line (crop -> photo pixels) -> RoofLine
      -> conservative geometry: snap, vanishing-point facade, neighbour-plane filter, corners
      -> occlusion against SAM's blockers -> confidence -> the editor's region rows

The roof lines come from ONE source: the first in ROOFLINE_SOURCES that answers. With
"v3,partition" that is V3; SAM's planes and blockers only give its lines context (crop, plane
filter, occlusion, confidence), and partition's own lines are never mixed in. Partition draws the
lines only when it is the configured source, or as the fallback when V3 fails.

Rules this module keeps:

- Geometry refines V3, it does not redraw it. Snapping moves an endpoint only on strong, local,
  roof-consistent evidence; vanishing points and facades are metadata and never move a line; the
  plane filter rejects only on strong evidence of a neighbour's plane. Every V3 line keeps its raw
  answer, and with ROOFLINE_DEBUG every stage is recorded so the first stage that damaged a good
  V3 line can be found (Recorder).
- V3 failing (no key, timeout, HTTP error, bad JSON, no lines) is a status, never a failed
  detection: the partition lines are used exactly as the partition-only pipeline draws them.
- ROOFLINE_SOURCES=partition is the existing pipeline, unchanged, plus confidence and sources.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from app.providers import roofline_v3
from app.services import occlusion, partition, snap, target
from app.services.rooflines import (
    CropBox, RoofLine, grow_to, normalize_partition_line, normalize_v3_line, overlap_share, target_crop,
)

log = logging.getLogger(__name__)

SAMPLE_STEP = 2.0
# Snapping: the share of the line nearest each end that decides that end, and the most the line
# may turn as a result. A roof edge is straight; a snap that bends it is following something else.
SNAP_END_SHARE = 0.35
SNAP_MIN_SAMPLES = 6
SNAP_MIN_LENGTH = 12.0          # px: shorter lines have too little edge to read
SNAP_MAX_TURN_DEG = 2.5
SNAP_AGREE_PX = 1.5             # a sample agrees with the snap when its own best edge is this close
SNAP_ALREADY_ON = 0.9           # the line already sits on an edge this strong (share of the best)
# Plane filter: a V3 line is a neighbour's when this much of it lies on excluded planes and
# almost none on the house's, both judged within a tolerance, never pixel-exact.
PLANE_TOLERANCE_L = 0.01
PLANE_EXCLUDED_SHARE = 0.6
PLANE_KEPT_SHARE = 0.25
# ... and only when this little of it lies within the house's columns: vertically in line with the
# kept roofs is the same house, only a line beside them can be a neighbour's.
PLANE_WITHIN_HOUSE_SHARE = 0.5
# Corners: reach, the least angle two lines need to make a corner, and how much a line may shrink.
CORNER_REACH_L = 0.012
CORNER_MIN_ANGLE_DEG = 15.0
CORNER_KEEP_LENGTH = 0.5
CORNER_BLOCKED = 0.5
# Facade: one horizontal vanishing point this far out (half image widths) faces the camera.
FRONT_VP_REACH = 3.0
# Confidence evidence.
EDGE_BAND_PX = 2.0
SOURCE_EVIDENCE = {"v3": 0.65, "partition": 0.5, "inferred": 0.3}
HORIZONTAL_TYPES = {"eave", "ridge", "junction"}
SLOPED_TYPES = {"rake", "hip", "valley"}


# -- V3 --------------------------------------------------------------------------------------------


@dataclass
class V3Outcome:
    """What V3 gave for a crop: "ok", "empty" (answered, no lines), "failed" or "skipped"."""

    status: str
    crop: CropBox | None = None
    raw_lines: list[dict] = field(default_factory=list)
    error: str | None = None
    meta: dict = field(default_factory=dict)
    # Where the crop came from: "target_house(_sky)", "whole_photo" (SAM found no roof planes),
    # "find_in_box", or "missed_plane_<n>" for a second call on a plane the first one missed.
    crop_basis: str | None = None
    # Second calls on planes this call missed (V3_MISSED_PLANES), each with its own crop.
    extra: list["V3Outcome"] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(self.raw_lines)

    def stored(self) -> dict:
        """What is kept on the photo so a House tap can reuse V3's answers without calling it again."""
        return {"status": self.status, "crop": self.crop.as_list() if self.crop else None,
                "crop_basis": self.crop_basis, "lines": self.raw_lines, "error": self.error,
                "extra": [call.stored() for call in self.extra]}

    @classmethod
    def from_stored(cls, data: dict | None) -> "V3Outcome | None":
        if not data or not data.get("crop"):
            return None
        extra = [call for call in (cls.from_stored(item) for item in data.get("extra") or []) if call is not None]
        return cls(status=data.get("status", "ok"), crop=CropBox(*data["crop"]), raw_lines=list(data.get("lines") or []),
                   error=data.get("error"), meta={"cache": "stored"}, crop_basis=data.get("crop_basis"), extra=extra)

    def calls(self) -> list[tuple[str, "V3Outcome"]]:
        """This call and its second calls, with the prefix their lines' ids carry (v3_001, v3m1_001)."""
        return [("v3", self)] + [(f"v3m{k}", call) for k, call in enumerate(self.extra, 1)]


def cache_dir(settings) -> Path | None:
    return Path(settings.storage_dir) / "v3_cache" if settings.v3_cache else None


def run_v3(image: Image.Image, crop: CropBox, settings, crop_basis: str = "target_house") -> V3Outcome:
    """V3 on one crop of the photo. Never raises: a failure is returned as a status."""
    if not settings.openrouter_api_key.strip():
        return V3Outcome("failed", crop, error="OPENROUTER_API_KEY is not set on the API server", crop_basis=crop_basis)
    try:
        lines, meta = roofline_v3.detect_crop(image, crop.as_list(), settings, cache_dir(settings))
    except RuntimeError as exc:
        # The provider's messages carry no response body or key (see roofline_v3.detect).
        log.warning("V3 unavailable, falling back to partition: %s", exc)
        return V3Outcome("failed", crop, error=str(exc), crop_basis=crop_basis)
    return V3Outcome("ok" if lines else "empty", crop, raw_lines=lines, meta=meta, crop_basis=crop_basis)


# Second calls: a crop around a missed plane is at least this big, so V3 has context; one that
# lies this much inside an earlier second call's crop adds nothing and is not made. The main crop
# is not compared: it holds every house plane, and a missed plane was in it. The second call is a
# closer look at what V3 had in view and still missed.
MISSED_MIN_CROP_PX = 192
MISSED_INSIDE_SHARE = 0.85


def plane_coverage(plane: np.ndarray, lines: list[RoofLine], blockers: np.ndarray | None, reach: float) -> float | None:
    """Share of a plane's visible outline lying within `reach` px of any of the lines, or None when
    no outline is visible (all of it behind objects)."""
    outline = plane & ~cv2.erode(plane.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if blockers is not None:
        outline &= ~blockers
    total = int(outline.sum())
    if not total:
        return None
    drawn = np.zeros(plane.shape, np.uint8)
    thickness = max(1, int(round(2 * reach + 1)))
    for line in lines:
        points = np.rint(np.asarray(line.points, float)).astype(np.int32)
        cv2.polylines(drawn, [points.reshape(-1, 1, 2)], False, 1, thickness)
    return float((outline & drawn.astype(bool)).sum()) / total


def run_missed_planes(image: Image.Image, v3: V3Outcome, roof_masks: list[np.ndarray], keep: list[bool] | None,
                      blockers: np.ndarray | None, sky: np.ndarray | None, settings) -> dict:
    """Ask V3 again, once per target-house plane its first answer missed (V3_MISSED_PLANES).

    A plane is missed when under V3_MISSED_COVERAGE of its visible outline lies near any line of
    the first answer. Planes under V3_MISSED_MIN_AREA of the house's roof area are ignored; the
    rest are taken largest first, each on its own crop (the plane's box, V3_MISSED_PADDING, sky-
    bounded top, at least MISSED_MIN_CROP_PX), skipping one whose crop is already mostly inside an
    earlier second call's, up to V3_MISSED_MAX_CALLS calls. Each call is cached and
    never raises. The answers go on `v3.extra`; the return value is the run's record.
    """
    report: dict = {"checked": 0, "missed": [], "calls": []}
    if not v3.usable or not roof_masks:
        return report
    size = image.size
    flags = keep if keep is not None else [True] * len(roof_masks)
    house = [i for i, (mask, k) in enumerate(zip(roof_masks, flags)) if k and mask.any()]
    if not house:
        return report
    extent = partition.roof_extent([roof_masks[i] for i in house])
    reach = max(6.0, PLANE_TOLERANCE_L * extent)
    first = [line for i, raw in enumerate(v3.raw_lines, 1)
             if (line := normalize_v3_line(raw, i, v3.crop, size)) is not None]
    house_area = float(sum(roof_masks[i].sum() for i in house))
    candidates = []
    for index in house:
        plane = roof_masks[index].astype(bool)
        report["checked"] += 1
        if float(plane.sum()) < settings.v3_missed_min_area * house_area:
            continue
        coverage = plane_coverage(plane, first, blockers, reach)
        if coverage is not None and coverage < settings.v3_missed_coverage:
            candidates.append((float(plane.sum()), index, coverage))
    done: list[CropBox] = []
    for area, index, coverage in sorted(candidates, reverse=True):
        entry = {"plane": index, "coverage": round(coverage, 3), "area": int(area)}
        report["missed"].append(entry)
        if len(v3.extra) >= settings.v3_missed_max_calls:
            entry["skipped"] = "max_calls"
            continue
        crop, _ = target_crop([roof_masks[index]], [True], size, settings.v3_missed_padding,
                              sky=sky if settings.v3_crop_sky else None, sky_max_rise=settings.v3_crop_sky_max_rise)
        crop = grow_to(crop, MISSED_MIN_CROP_PX, size)
        if any(overlap_share(crop, earlier) >= MISSED_INSIDE_SHARE for earlier in done):
            entry["skipped"] = "inside_earlier_crop"
            continue
        call = run_v3(image, crop, settings, crop_basis=f"missed_plane_{index}")
        done.append(crop)
        v3.extra.append(call)
        entry["call"] = len(v3.extra)
        report["calls"].append({"plane": index, "crop": crop.as_list(), "status": call.status, "error": call.error,
                                "lines": len(call.raw_lines), "cache": call.meta.get("cache"),
                                "cost_usd": call.meta.get("cost_usd"), "elapsed_seconds": call.meta.get("elapsed_seconds")})
    log.info("V3 missed planes: %d checked, %d missed, %d second calls", report["checked"], len(report["missed"]), len(v3.extra))
    return report


# -- context: everything the geometry reads, computed once --------------------------------------


class Context:
    """The photo, its planes and blockers, and the maps geometry and confidence read, lazily."""

    def __init__(self, image: Image.Image, settings, roof_masks: list[np.ndarray], keep: list[bool] | None,
                 blockers: np.ndarray | None, vp_region: np.ndarray | None = None, crop: CropBox | None = None):
        self.image = image
        self.settings = settings
        self.size = image.size
        self.width, self.height = image.size
        self.roof_masks = [m.astype(bool) for m in roof_masks]
        self.keep = keep
        self.blockers = blockers
        self.vp_region = vp_region
        self.crop = crop
        self.extent = partition.roof_extent(self.roof_masks) if self.roof_masks else float(max(image.size))
        self._maps: dict = {}

    @property
    def gradients(self):
        if "gradients" not in self._maps:
            self._maps["gradients"] = snap.field(self.image)
        return self._maps["gradients"]

    @property
    def roof_union(self) -> np.ndarray | None:
        if not self.roof_masks:
            return None
        if "union" not in self._maps:
            self._maps["union"] = np.logical_or.reduce(self.roof_masks)
        return self._maps["union"]

    def _distance(self, name: str, mask: np.ndarray | None) -> np.ndarray | None:
        """Distance in px from every pixel to the nearest pixel of `mask` (None without one)."""
        if name not in self._maps:
            if mask is None or not mask.any():
                self._maps[name] = None
            else:
                self._maps[name] = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        return self._maps[name]

    def roof_distance(self) -> np.ndarray | None:
        return self._distance("roof", self.roof_union)

    def plane_masks(self, kept: bool) -> np.ndarray | None:
        if not self.roof_masks:
            return None
        flags = self.keep if self.keep is not None else [True] * len(self.roof_masks)
        chosen = [m for m, k in zip(self.roof_masks, flags) if bool(k) == kept]
        return np.logical_or.reduce(chosen) if chosen else None

    def house_columns(self) -> tuple[int, int] | None:
        """The left-right span (first, last column) of the house's kept planes, or None."""
        if "columns" not in self._maps:
            kept = self.plane_masks(True)
            columns = np.flatnonzero(kept.any(axis=0)) if kept is not None else np.zeros(0, int)
            self._maps["columns"] = (int(columns[0]), int(columns[-1])) if columns.size else None
        return self._maps["columns"]

    def kept_distance(self) -> np.ndarray | None:
        return self._distance("kept", self.plane_masks(True))

    def excluded_distance(self) -> np.ndarray | None:
        return self._distance("excluded", self.plane_masks(False))

    def boundary_distance(self) -> np.ndarray | None:
        """Distance to the outline of any house plane: where roof edges, hips and ridges lie."""
        if "boundary" not in self._maps:
            flags = self.keep if self.keep is not None else [True] * len(self.roof_masks)
            outline = np.zeros((self.height, self.width), bool)
            kernel = np.ones((3, 3), np.uint8)
            for mask, keep_it in zip(self.roof_masks, flags):
                if keep_it and mask.any():
                    outline |= mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            return self._distance("boundary", outline if outline.any() else None)
        return self._maps["boundary"]

    @property
    def scene(self) -> partition.Scene | None:
        """The trusted vanishing points: the photo's, refit to the target house's own segments."""
        if "scene" not in self._maps:
            self._maps["scene"] = self._scene()
        return self._maps["scene"]

    def _scene(self) -> partition.Scene | None:
        rgb = np.asarray(self.image.convert("RGB"))
        long_edge = float(max(self.size))
        segments = partition.lsd_segments(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), partition.foliage_mask(rgb, long_edge))
        scene = partition.vanishing_points(segments, (self.height, self.width))
        if not len(segments) or (scene.vertical is None and not scene.horizontal):
            return scene
        # Only the target house's segments decide where its vanishing points are (m0's rule),
        # narrowed to the house's crop when there is one.
        region = self.vp_region.astype(bool) if self.vp_region is not None else np.ones((self.height, self.width), bool)
        if self.crop is not None:
            inside = np.zeros_like(region)
            inside[self.crop.y0:self.crop.y1, self.crop.x0:self.crop.x1] = True
            region = region & inside
        counts = partition._segments_within(segments, region)
        if counts.any() and not counts.all() and self.roof_masks:
            house = [i for i, keep_it in enumerate(self.keep or [True] * len(self.roof_masks)) if keep_it]
            scene = partition.house_vanishing_points(
                scene, counts, partition._house_probes(self.roof_masks, house, scene.centre, scene.scale))
        return scene

    def samples(self, line: RoofLine) -> np.ndarray:
        coords, _ = occlusion._samples(np.asarray(line.points, float))
        return coords

    def lookup(self, image_map: np.ndarray, coords: np.ndarray) -> np.ndarray:
        xs = np.clip(np.rint(coords[:, 0]).astype(int), 0, self.width - 1)
        ys = np.clip(np.rint(coords[:, 1]).astype(int), 0, self.height - 1)
        return image_map[ys, xs]


def _bilinear(image_map: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image_map.shape
    xs = np.clip(xs, 0.0, width - 1.001)
    ys = np.clip(ys, 0.0, height - 1.001)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    return (image_map[y0, x0] * (1 - fx) * (1 - fy) + image_map[y0, x0 + 1] * fx * (1 - fy)
            + image_map[y0 + 1, x0] * (1 - fx) * fy + image_map[y0 + 1, x0 + 1] * fx * fy)


def _across(ctx: Context, a: np.ndarray, b: np.ndarray, offsets: np.ndarray, step: float = SAMPLE_STEP):
    """Gradient strength across the line a->b, at every sample along it and every offset along
    its normal: (samples, offsets) array, the along-line positions t in [0, 1], and the normal."""
    gx, gy, _ = ctx.gradients
    direction = b - a
    length = float(np.linalg.norm(direction))
    along = direction / length
    normal = np.array([-along[1], along[0]])
    t = np.arange(0.0, length + 1e-9, step) / length
    base = a[None, :] + t[:, None] * direction[None, :]
    xs = base[:, 0, None] + normal[0] * offsets[None, :]
    ys = base[:, 1, None] + normal[1] * offsets[None, :]
    values = np.abs(_bilinear(gx, xs, ys) * normal[0] + _bilinear(gy, xs, ys) * normal[1])
    return values, t, normal


# -- geometry: conservative snap -------------------------------------------------------------------


def snap_line(line: RoofLine, ctx: Context) -> RoofLine:
    """Move each end of a V3 line onto a strong parallel image edge nearby, only on good evidence.

    Each end is judged on the part of the line nearest it: the mean gradient across the line at
    every offset within the radius gives a profile; its peak is the candidate. The candidate is
    taken only when it is strong (against the photo's own strong edges), consistent (most samples
    put their own best edge there too: a straight edge, not a window corner), near (a distance
    penalty), on the roof side of things (never further from the roof masks, never onto a blocker),
    and the line does not turn by more than SNAP_MAX_TURN_DEG. Otherwise the end stays exactly
    where V3 put it. Side, type and the raw answer never change here.
    """
    line.evidence.setdefault("pre_snap", [list(map(float, p)) for p in line.points])
    if len(line.points) != 2 or line.length < SNAP_MIN_LENGTH:
        line.evidence["snap"] = {"reason": "too_short"}
        return line
    settings = ctx.settings
    radius = max(4.0, settings.v3_snap_radius * ctx.extent)
    offsets = np.arange(-radius, radius + 0.25, 0.5)
    a, b = np.asarray(line.points[0], float), np.asarray(line.points[1], float)
    values, t, normal = _across(ctx, a, b, offsets)
    strong = ctx.gradients[2]
    zero = int(np.argmin(np.abs(offsets)))
    roof_distance = ctx.roof_distance()
    ends = []
    for index, anchor in ((0, a), (1, b)):
        window = t <= SNAP_END_SHARE if index == 0 else t >= 1 - SNAP_END_SHARE
        if window.sum() < SNAP_MIN_SAMPLES:
            window = np.argsort(t if index == 0 else -t)[:SNAP_MIN_SAMPLES]
        block = values[window]
        profile = block.mean(axis=0)
        weighted = profile * (1.0 - 0.25 * np.abs(offsets) / radius)
        best = int(np.argmax(weighted))
        if profile[zero] >= SNAP_ALREADY_ON * profile[best]:
            best = zero  # already on the edge: nothing to gain by moving
        offset = float(offsets[best])
        own = offsets[np.argmax(block, axis=1)]
        consistency = float(np.mean(np.abs(own - offset) <= SNAP_AGREE_PX))
        strength = float(min(1.0, profile[best] / strong))
        quality = strength * consistency * (1.0 - 0.5 * abs(offset) / radius)
        moved = anchor + normal * offset
        reason = None
        if abs(offset) < 0.75:
            reason = "on_edge" if best == zero else "negligible"
        elif quality < settings.v3_snap_min_quality:
            reason = "weak_evidence"
        elif ctx.blockers is not None and bool(ctx.lookup(ctx.blockers, moved[None, :])[0]):
            reason = "onto_blocker"
        elif roof_distance is not None:
            before = float(ctx.lookup(roof_distance, anchor[None, :])[0])
            after = float(ctx.lookup(roof_distance, moved[None, :])[0])
            if after > before + 1.5 and after > PLANE_TOLERANCE_L * ctx.extent:
                reason = "away_from_roof"
        ends.append({"offset": offset if reason is None else 0.0, "candidate": round(offset, 2), "quality": round(quality, 3),
                     "strength": round(strength, 3), "consistency": round(consistency, 3), "reason": reason or "snapped"})
    # A snap may not turn the line: keep the better end's move, drop the other's.
    moved_a, moved_b = a + normal * ends[0]["offset"], b + normal * ends[1]["offset"]
    turn = partition._angle_between(moved_b - moved_a, b - a)
    if turn > SNAP_MAX_TURN_DEG:
        weaker = 0 if ends[0]["quality"] < ends[1]["quality"] else 1
        ends[weaker].update(offset=0.0, reason="would_turn_line")
        moved_a, moved_b = a + normal * ends[0]["offset"], b + normal * ends[1]["offset"]
    distance = max(abs(ends[0]["offset"]), abs(ends[1]["offset"]))
    line.snap_quality = round(float(np.mean([e["quality"] for e in ends])), 3)
    line.evidence["snap"] = {"radius": round(radius, 1), "ends": ends}
    if distance > 0:
        line.points = [moved_a.tolist(), moved_b.tolist()]
        line.snapped = True
        line.snap_distance = round(distance, 2)
    return line


# -- geometry: vanishing points and facades (metadata only) ---------------------------------------


def _vp_image(vp: np.ndarray, scene: partition.Scene) -> tuple[float, float] | None:
    if abs(vp[2]) < 1e-9:
        return None
    point = vp[:2] / vp[2] * scene.scale + scene.centre
    return float(point[0]), float(point[1])


def _facades(scene: partition.Scene) -> list[str]:
    """Per horizontal vanishing point, the facade whose level lines run to it.

    The facade facing the camera most squarely has near-parallel level lines: its vanishing point
    is the farthest out (or at infinity). That one is "front", the project's definition of the
    front. A facade receding to the left has its vanishing point left of the photo's centre.
    """
    positions = []
    for vp in scene.horizontal:
        positions.append(np.inf if abs(vp[2]) < 1e-9 else float(np.hypot(*(vp[:2] / vp[2]))))
    facades = []
    for vp, reach in zip(scene.horizontal, positions):
        x = 0.0 if abs(vp[2]) < 1e-9 else float(vp[0] / vp[2])
        if len(scene.horizontal) >= 2:
            front = reach == max(positions)
        else:
            front = reach >= FRONT_VP_REACH
        facades.append("front" if front else ("left" if x < 0 else "right"))
    return facades


def assign_vanishing_point(line: RoofLine, ctx: Context) -> RoofLine:
    """Extend the line, and associate it with the trusted vanishing point it runs toward.

    Measured as the angle between the line and the direction from its middle to the vanishing
    point, never from the line's image slope alone. Sets vanishing_point, vp_angle, facade; the
    coordinates are not touched.
    """
    scene = ctx.scene
    if scene is None or not scene.horizontal and scene.vertical is None:
        line.evidence["vp"] = "no_scene"
        return line
    middle = (np.asarray(line.points[0], float) + np.asarray(line.points[-1], float)) / 2
    normalised = (middle - scene.centre) / scene.scale
    direction = line.direction
    if scene.vertical is not None:
        vertical = partition._angle_between(direction, partition._vp_direction(scene.vertical, normalised))
        line.evidence["vertical_angle"] = round(vertical, 2)
        if vertical <= ctx.settings.roofline_vp_deg:
            line.evidence["vp"] = "vertical"  # runs toward the vertical VP: a wall edge, not level
    facades = _facades(scene)
    best = None
    for index, vp in enumerate(scene.horizontal):
        angle = partition._angle_between(direction, partition._vp_direction(vp, normalised))
        if angle <= ctx.settings.roofline_vp_deg and (best is None or angle < best[0]):
            best = (angle, index)
    if best is not None:
        angle, index = best
        line.vp_angle = round(angle, 2)
        line.vanishing_point = _vp_image(scene.horizontal[index], scene)
        line.facade = facades[index]
        line.facade_basis = "vp"
        line.evidence["vp"] = f"horizontal_{index}"
        line.evidence["vp_index"] = index
    elif "vp" not in line.evidence:
        line.evidence["vp"] = "none"
    return line


def assign_facade_by_adjacency(lines: list[RoofLine], ctx: Context) -> None:
    """A line with no vanishing point of its own (a rake) takes the facade of the level lines it
    meets, when they all agree. A hip meets two facades and so gets none. One pass, no spreading."""
    reach = max(4.0, CORNER_REACH_L * ctx.extent)
    anchored = [l for l in lines if l.facade_basis == "vp"]
    for line in lines:
        if line.facade is not None:
            continue
        found = set()
        for other in anchored:
            if other is line:
                continue
            for end in (line.points[0], line.points[-1]):
                for tip in (other.points[0], other.points[-1]):
                    if np.hypot(end[0] - tip[0], end[1] - tip[1]) <= reach:
                        found.add(other.facade)
        if len(found) == 1:
            line.facade = found.pop()
            line.facade_basis = "adjacent"


# -- geometry: plane filter, corners ---------------------------------------------------------------


def plane_filter(line: RoofLine, ctx: Context) -> bool:
    """Keep a V3 line unless it plainly lies on a neighbour's roof.

    Tolerant by design: V3 sees the photo, SAM's masks are approximate, and a real roof edge can
    sit several pixels outside a mask. A line is rejected only when it lies mostly on excluded
    planes, hardly on the house's, AND mostly outside the house's columns (the left-right span of
    its kept planes, within the tolerance). Neighbours stand beside the house; a line above or
    below its roofs is the same house even when the selection left that plane out (an upper
    storey, a turret). A line near no plane at all (a roof SAM missed) is kept.
    """
    excluded = ctx.excluded_distance()
    kept = ctx.kept_distance()
    if excluded is None:
        line.plane_status = "kept:no_excluded_planes"
        return True
    tolerance = max(6.0, PLANE_TOLERANCE_L * ctx.extent)
    coords = ctx.samples(line)
    on_excluded = float(np.mean(ctx.lookup(excluded, coords) <= tolerance))
    to_kept = ctx.lookup(kept, coords) if kept is not None else np.full(len(coords), np.inf)
    on_kept = float(np.mean(to_kept <= tolerance))
    line.evidence["plane"] = {"on_excluded": round(on_excluded, 3), "on_kept": round(on_kept, 3),
                              "distance_to_mask": round(float(np.median(to_kept)), 1) if np.isfinite(to_kept).any() else None}
    if on_excluded >= PLANE_EXCLUDED_SHARE and on_kept < PLANE_KEPT_SHARE:
        columns = ctx.house_columns()
        within = float(np.mean((coords[:, 0] >= columns[0] - tolerance) & (coords[:, 0] <= columns[1] + tolerance)))             if columns is not None else 0.0
        line.evidence["plane"]["within_house_width"] = round(within, 3)
        if within >= PLANE_WITHIN_HOUSE_SHARE:
            line.plane_status = "kept:within_house_width"
            return True
        line.plane_status = "rejected:excluded_plane"
        log.info("%s rejected: reason=excluded_plane distance_to_mask=%s on_excluded=%.2f",
                 line.source_id, line.evidence["plane"]["distance_to_mask"], on_excluded)
        return False
    line.plane_status = "kept"
    return True


def _intersection(a: RoofLine, b: RoofLine) -> np.ndarray | None:
    p, r = np.asarray(a.points[0], float), np.asarray(a.points[-1], float) - np.asarray(a.points[0], float)
    q, s = np.asarray(b.points[0], float), np.asarray(b.points[-1], float) - np.asarray(b.points[0], float)
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        return None
    u = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / cross
    return p + u * r


def close_corners(lines: list[RoofLine], ctx: Context) -> int:
    """Join ends of two V3 lines that meet at a corner: only when the ends are close, the lines
    really cross there (not parallel, not two lines to one vanishing point), no object sits in the
    corner, and neither line loses more than half its length. Returns the corners closed."""
    reach = max(4.0, CORNER_REACH_L * ctx.extent)
    candidates = []
    for i, first in enumerate(lines):
        for j in range(i + 1, len(lines)):
            second = lines[j]
            if partition._angle_between(first.direction, second.direction) < CORNER_MIN_ANGLE_DEG:
                continue
            same_vp = first.evidence.get("vp_index") is not None and first.evidence.get("vp_index") == second.evidence.get("vp_index")
            if same_vp:
                continue
            for end_i in (0, -1):
                for end_j in (0, -1):
                    gap = float(np.hypot(first.points[end_i][0] - second.points[end_j][0],
                                         first.points[end_i][1] - second.points[end_j][1]))
                    if 0.5 < gap <= reach:
                        candidates.append((gap, i, end_i, j, end_j))
    used: set = set()
    closed = 0
    for gap, i, end_i, j, end_j in sorted(candidates):
        if (i, end_i) in used or (j, end_j) in used:
            continue
        first, second = lines[i], lines[j]
        corner = _intersection(first, second)
        if corner is None:
            continue
        a, b = np.asarray(first.points[end_i], float), np.asarray(second.points[end_j], float)
        if np.linalg.norm(corner - a) > reach or np.linalg.norm(corner - b) > reach:
            continue
        if ctx.blockers is not None:
            covered = [occlusion._coverage(ctx.blockers, end, corner) for end in (a, b) if np.linalg.norm(corner - end) >= 1]
            if any(c >= CORNER_BLOCKED for c in covered):
                continue
        before = (first.length, second.length)
        trial_first, trial_second = [list(p) for p in first.points], [list(p) for p in second.points]
        trial_first[end_i], trial_second[end_j] = corner.tolist(), corner.tolist()
        lengths = [float(np.linalg.norm(np.diff(np.asarray(t, float), axis=0), axis=1).sum()) for t in (trial_first, trial_second)]
        if lengths[0] < CORNER_KEEP_LENGTH * before[0] or lengths[1] < CORNER_KEEP_LENGTH * before[1]:
            continue
        first.points, second.points = trial_first, trial_second
        for line, partner, end in ((first, second, end_i), (second, first, end_j)):
            line.evidence.setdefault("corners", []).append({"with": partner.source_id, "end": end, "moved": round(gap / 2, 2)})
        used |= {(i, end_i), (j, end_j)}
        closed += 1
    return closed


# -- matching: never add a line that is already there --------------------------------------------


def _types_compatible(a: str | None, b: str | None) -> bool:
    """A junction (roof meeting wall) is its own thing; the structural edge types may disagree on
    what an edge is (V3's rake, partition's hip) and still be one edge; unknown matches anything."""
    if a in (None, "unknown") or b in (None, "unknown") or a == b:
        return True
    return "junction" not in (a, b)


def relation(a: RoofLine, b: RoofLine) -> dict:
    """How two lines sit relative to each other: angle, perpendicular distance, overlap share."""
    angle = partition._angle_between(a.direction, b.direction)
    longer, shorter = (a, b) if a.length >= b.length else (b, a)
    origin = np.asarray(longer.points[0], float)
    along = longer.direction
    normal = np.array([-along[1], along[0]])
    ends = np.asarray([shorter.points[0], shorter.points[-1]], float)
    distance = float(np.abs((ends - origin) @ normal).max())
    span = sorted([0.0, float((np.asarray(longer.points[-1], float) - origin) @ along)])
    at = sorted([float((ends[0] - origin) @ along), float((ends[1] - origin) @ along)])
    shared = max(0.0, min(span[1], at[1]) - max(span[0], at[0]))
    overlap = shared / max(shorter.length, 1e-6)
    return {"angle": angle, "distance": distance, "overlap": overlap}


def _edge_support(line: RoofLine, ctx: Context) -> float | None:
    """Mean gradient across the line within EDGE_BAND_PX, over its visible samples, against the
    photo's own strong edges, in [0, 1]. None when all of it is behind objects."""
    array = np.asarray(line.points, float)
    offsets = np.arange(-EDGE_BAND_PX, EDGE_BAND_PX + 0.25, 0.5)
    strong = ctx.gradients[2]
    scores, weights = [], []
    for a, b in zip(array[:-1], array[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1.0:
            continue
        values, t, _ = _across(ctx, a, b, offsets)
        best = values.max(axis=1)
        points = a[None, :] + t[:, None] * (b - a)[None, :]
        visible = ~ctx.lookup(ctx.blockers, points) if ctx.blockers is not None else np.ones(len(points), bool)
        if visible.any():
            scores.append(float(np.minimum(1.0, best[visible] / strong).mean()))
            weights.append(float(visible.sum()))
    if not scores:
        return None
    return float(np.average(scores, weights=weights))


def dedupe_against(candidates: list[RoofLine], existing: list[RoofLine], ctx: Context) -> list[RoofLine]:
    """Candidates that do not repeat an existing line: Find in Box against the photo's lines, a
    second V3 call against the first call's. The existing lines are never changed."""
    limit = max(5.0, ctx.settings.roofline_match_distance * ctx.extent)
    kept = []
    for line in candidates:
        repeats = False
        for other in existing:
            if not _types_compatible(line.line_type, other.line_type):
                continue
            r = relation(line, other)
            if r["angle"] <= ctx.settings.roofline_match_angle_deg and r["distance"] <= limit and r["overlap"] >= ctx.settings.roofline_match_overlap:
                repeats = True
                line.evidence["duplicate_of"] = other.source_id
                break
        if not repeats:
            kept.append(line)
    return kept


# -- occlusion on the source's lines ---------------------------------------------------------------------


def handle_occlusion(lines: list[RoofLine], ctx: Context) -> tuple[list[RoofLine], dict]:
    """The existing bridge-and-extend rules (partition.bridge_through), on V3's lines.

    Same-type pieces that line up across a detected object are joined into one straight line
    (occlusion._gap_if_continues decides, with partition's cover and offset rules); an end an
    object cuts off is carried to its far side only where the roof masks show roof
    (occlusion.extend_behind). Collinearity alone never joins: the gap must be covered. A join
    lists both pieces.
    """
    counts = {"input": len(lines), "bridged": 0, "extended": 0}
    blockers = ctx.blockers
    if blockers is None:
        counts["output"] = len(lines)
        return list(lines), counts
    extent = ctx.extent
    cover = cv2.dilate(blockers.astype(np.uint8), partition._ellipse(partition.BLOCKER_DILATE_L * extent)).astype(bool)
    roof = ctx.roof_union
    if roof is not None:
        grow = 2 * partition.BRIDGE_ROOF_GROW_PX + 1
        cover |= cv2.dilate(roof.astype(np.uint8), np.ones((grow, grow), np.uint8)).astype(bool)
    max_offset = partition.BRIDGE_OFFSET_L * extent
    pieces = list(lines)
    while True:
        best = None
        for i, left in enumerate(pieces):
            a = np.asarray(left.points, float)
            if not occlusion._rightward(a):
                continue
            for j, right in enumerate(pieces):
                if i == j or right.line_type != left.line_type:
                    continue
                b = np.asarray(right.points, float)
                if not occlusion._rightward(b):
                    continue
                gap = occlusion._gap_if_continues(a, b, cover, ctx.width, max_offset)
                if gap is not None and (best is None or gap < best[0]):
                    best = (gap, i, j)
        if best is None:
            break
        _, i, j = best
        left, right = pieces[i], pieces[j]
        joined = (left if left.length >= right.length else right).clone()
        points = np.vstack([np.asarray(left.points, float), np.asarray(right.points, float)])
        chord = points[-1] - points[0]
        norm = float(np.linalg.norm(chord))
        if norm > 1e-6:
            normal = np.array([-chord[1], chord[0]]) / norm
            if float(np.abs((points - points[0]) @ normal).max()) <= partition.BRIDGE_STRAIGHT_L * extent:
                points = points[[0, -1]]
        joined.points = points.tolist()
        joined.source = list(dict.fromkeys(left.source + right.source))
        joined.bridged_from = [s for s in (left.bridged_from or [left.source_id]) + (right.bridged_from or [right.source_id]) if s]
        joined.occluded = True
        joined.inferred = left.inferred and right.inferred
        joined.snapped = left.snapped or right.snapped
        # Side belongs to the facade or plane, never to the joined line's slope.
        joined.side = left.side if left.side == right.side else (joined.facade or left.side or right.side)
        pieces = [p for k, p in enumerate(pieces) if k not in (i, j)] + [joined]
        counts["bridged"] += 1
    out = []
    for line in pieces:
        if roof is not None:
            carried = occlusion.extend_behind([(line.int_points(), line.side, line.line_type)], blockers, extent, roof)[0][0]
            if carried != line.int_points():
                line.points = [list(map(float, p)) for p in carried]
                line.occluded = True
                line.evidence["extended_behind"] = True
                counts["extended"] += 1
        out.append(line)
    counts["output"] = len(out)
    return out, counts


# -- confidence ----------------------------------------------------------------------------------------


def calculate_confidence(line: RoofLine, lines: list[RoofLine], ctx: Context) -> float:
    """Confidence in [0, 1] from measured evidence, weighted by Settings.roofline_confidence_weights
    over the components that apply to this line. Deterministic for the same inputs.

    - source: V3 > partition > partition's inferred connector (drawn from no pixels);
    - edge: how strongly the photo shows an edge along the visible part of the line;
    - vp: how closely a level line (eave, ridge, junction) runs to a trusted vanishing point, or
      whether a sloped one has a facade by adjacency; not applicable without vanishing points;
    - geometry: ends meeting other lines, and length against the roof's size;
    - plane: how much of the line lies on the outline of the house's planes;
    - occlusion: how much of the line is actually visible.
    The weights are heuristics to be validated on the annotated photos, not a calibrated probability.
    """
    evidence: dict[str, float | None] = {}
    if "v3" in line.source:
        evidence["source"] = SOURCE_EVIDENCE["v3"]
    else:
        evidence["source"] = SOURCE_EVIDENCE["inferred" if line.inferred else "partition"]

    evidence["edge"] = _edge_support(line, ctx)

    scene = ctx.scene
    if scene is None or not scene.horizontal:
        evidence["vp"] = None
    elif line.vp_angle is not None:
        evidence["vp"] = max(0.0, 1.0 - line.vp_angle / max(ctx.settings.roofline_vp_deg, 1e-6))
    elif line.line_type in HORIZONTAL_TYPES:
        # A level edge that runs to no level vanishing point is weak; one that runs to the
        # vertical vanishing point is a wall edge, not a roof edge.
        evidence["vp"] = 0.0 if line.evidence.get("vp") == "vertical" else 0.2
    elif line.line_type in SLOPED_TYPES:
        evidence["vp"] = 0.6 if line.facade_basis == "adjacent" else None
    else:
        evidence["vp"] = 0.3

    reach = max(4.0, CORNER_REACH_L * ctx.extent)
    connected = 0
    for end in (line.points[0], line.points[-1]):
        point = np.asarray(end, float)
        for other in lines:
            if other is line:
                continue
            array = np.asarray(other.points, float)
            near = min(float(np.min(np.linalg.norm(array - point, axis=1))),
                       min((_point_segment(point, p, q) for p, q in zip(array[:-1], array[1:])), default=np.inf))
            if near <= reach:
                connected += 1
                break
    evidence["geometry"] = 0.6 * connected / 2 + 0.4 * min(1.0, line.length / max(0.04 * ctx.extent, 1.0))

    boundary = ctx.boundary_distance()
    coords = ctx.samples(line)
    if boundary is not None:
        tolerance = max(6.0, PLANE_TOLERANCE_L * ctx.extent)
        evidence["plane"] = float(np.mean(ctx.lookup(boundary, coords) <= tolerance))
    else:
        evidence["plane"] = None

    if ctx.blockers is not None:
        hidden = float(np.mean(ctx.lookup(ctx.blockers, coords)))
        evidence["occlusion"] = 0.9 if line.bridged_from and hidden > 0 else 1.0 - 0.6 * hidden
    else:
        evidence["occlusion"] = 1.0

    weights = ctx.settings.roofline_confidence_weights
    total = sum(weights.get(k, 0.0) for k, v in evidence.items() if v is not None)
    score = sum(weights.get(k, 0.0) * v for k, v in evidence.items() if v is not None) / total if total else 0.0
    line.evidence["confidence"] = {k: (round(v, 3) if v is not None else None) for k, v in evidence.items()}
    line.confidence = round(float(min(1.0, max(0.0, score))), 3)
    return line.confidence


def _point_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denominator = float(ab @ ab)
    t = 0.0 if denominator == 0 else min(1.0, max(0.0, float((point - a) @ ab) / denominator))
    return float(np.linalg.norm(point - (a + t * ab)))


def level(confidence: float | None, settings) -> str | None:
    if confidence is None:
        return None
    if confidence >= settings.roofline_confidence_high:
        return "high"
    return "low" if confidence < settings.roofline_confidence_low else "medium"


# -- instrumentation -------------------------------------------------------------------------------------


class Recorder:
    """Counts for every run; with ROOFLINE_DEBUG, every stage's lines and each V3 line's lifecycle.

    The lifecycle answers "was this line wrong when V3 produced it, or did a stage make it
    wrong?": A raw V3 -> B coordinates -> C snap -> D vanishing points -> E plane filter ->
    F corners -> H occlusion -> I final. P holds partition's lines when partition is the source.
    """

    STAGES = ("A_raw_v3", "B_converted", "C_snapped", "D_vp", "E_plane_filter", "F_corners",
              "H_occlusion", "P_partition", "I_final")

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.counts: dict = {}
        self.stages: dict[str, list] = {}
        self.records: dict[str, dict] = {}

    def stage(self, name: str, lines) -> None:
        if not self.enabled:
            return
        self.stages[name] = [line if isinstance(line, dict) else _brief(line) for line in lines]

    def note(self, line_id: str | None, **values) -> None:
        if self.enabled and line_id:
            self.records.setdefault(line_id, {"id": line_id}).update(values)

    def export(self) -> dict | None:
        if not self.enabled:
            return None
        return {"stages": self.stages, "lines": list(self.records.values())}


def _brief(line: RoofLine) -> dict:
    return {"id": line.source_id, "points": [[round(float(x), 1), round(float(y), 1)] for x, y in line.points],
            "type": line.line_type, "source": list(line.source), "side": line.side, "facade": line.facade,
            "snapped": line.snapped, "occluded": line.occluded, "confidence": line.confidence}


# -- the pipeline ------------------------------------------------------------------------------------------


@dataclass
class Result:
    lines: list[RoofLine]            # final, in the order the editor lists them
    requested: list[bool]            # per line: lit by the instruction
    inferred: set                    # partition's inferred connector keys among the final lines
    details: dict                    # partition's details (pitch, planes)
    report: dict                     # photo.interpreted.roofline
    debug: dict | None               # photo.interpreted.roofline_debug (ROOFLINE_DEBUG only)
    scene: partition.Scene | None = None  # the target house's vanishing points (services/camera.py)


PartitionLines = Callable[[object], tuple[list[tuple], list[tuple], dict]]


def refine_v3(outcome: V3Outcome, ctx: Context, recorder: Recorder, *, filter_planes: bool = True) -> list[RoofLine]:
    """Stages A-F for V3's lines: normalise into photo pixels, snap, vanishing points, plane
    filter, corners. Each stage is recorded; a line's raw answer is never lost."""
    counts = recorder.counts.setdefault("v3", {})
    calls = [(prefix, call) for prefix, call in outcome.calls() if call.usable]
    counts["raw_lines"] = sum(len(call.raw_lines) for _, call in calls)
    recorder.stage("A_raw_v3", [{"id": f"{prefix}_{i:03d}", "type": raw.get("type"), "crop_points": raw.get("points"),
                                 "crop": call.crop.as_list()}
                                for prefix, call in calls for i, raw in enumerate(call.raw_lines, 1)])
    lines: list[RoofLine] = []
    for number, (prefix, call) in enumerate(calls):
        found = []
        for index, raw in enumerate(call.raw_lines, 1):
            # Each call's lines are mapped back through that call's own crop.
            line = normalize_v3_line(raw, index, call.crop, ctx.size, prefix=prefix)
            recorder.note(f"{prefix}_{index:03d}", raw=True, raw_type=raw.get("type"), raw_points=raw.get("points"),
                          crop=call.crop.as_list(), coordinate_converted=line is not None,
                          **({} if line else {"final": False, "dropped_at": "B_converted"}))
            if line is not None:
                found.append(line)
        if number:
            # A second call's crop overlaps what the first call saw: its repeats are not added.
            kept = dedupe_against(found, lines, ctx)
            second = counts.setdefault("missed_planes", {"calls": 0, "lines": 0, "duplicates": 0})
            second["calls"] += 1
            second["lines"] += len(kept)
            second["duplicates"] += len(found) - len(kept)
            survivors = {id(line) for line in kept}
            for line in found:
                if id(line) not in survivors:
                    recorder.note(line.source_id, final=False, dropped_at="B_duplicate_of_first_call",
                                  duplicate_of=line.evidence.get("duplicate_of"))
            found = kept
        lines += found
    counts["valid_lines"] = counts["after_coordinate_conversion"] = len(lines)
    recorder.stage("B_converted", lines)

    if ctx.settings.v3_snap:
        for line in lines:
            snap_line(line, ctx)
            recorder.note(line.source_id, snapped=line.snapped, snap_distance=line.snap_distance, snap_quality=line.snap_quality)
    counts["after_snap"] = len(lines)
    counts["snapped"] = sum(1 for l in lines if l.snapped)
    recorder.stage("C_snapped", lines)

    for line in lines:
        assign_vanishing_point(line, ctx)
    assign_facade_by_adjacency(lines, ctx)
    for line in lines:
        line.side = line.facade
        recorder.note(line.source_id, vp_assigned=line.vanishing_point is not None or line.facade_basis == "vp",
                      facade=line.facade, facade_basis=line.facade_basis)
    counts["after_vp_classification"] = len(lines)
    counts["vp_assigned"] = sum(1 for l in lines if l.facade_basis == "vp")
    recorder.stage("D_vp", lines)

    kept = []
    for line in lines:
        keep_it = plane_filter(line, ctx) if filter_planes else True
        if not filter_planes:
            line.plane_status = "kept:not_applied"
        recorder.note(line.source_id, plane_filter=line.plane_status, **({} if keep_it else {"final": False, "dropped_at": "E_plane_filter"}))
        if keep_it:
            kept.append(line)
    counts["after_plane_filter"] = len(kept)
    counts["rejected_excluded_plane"] = len(lines) - len(kept)
    recorder.stage("E_plane_filter", kept)

    counts["corners_closed"] = close_corners(kept, ctx)
    recorder.stage("F_corners", kept)
    return kept


def unify(
    *,
    image: Image.Image,
    settings,
    parsed,
    roof_masks: list[np.ndarray],
    keep: list[bool] | None,
    blockers: np.ndarray | None,
    partition_lines: PartitionLines | None,
    v3: V3Outcome | None,
    vp_region: np.ndarray | None = None,
    crop: CropBox | None = None,
) -> Result:
    """The roof lines for one detection (or a House tap re-running it without V3 calls).

    Roof lines come from one source: V3 when it answered, else partition (the fallback, or the
    only source with ROOFLINE_SOURCES=partition). SAM's planes and blockers are context for them,
    never extra lines. `keep` is the target house's plane selection (None: every plane counts);
    `partition_lines(parsed) -> (spans, unrequested, details)` runs the partition geometry, already
    restricted to the house when TARGET_HOUSE_ONLY is on, and is only called when partition is the
    source; `v3` is V3's outcome for the crop, or None when V3 is not a source.
    """
    sources = list(settings.roofline_sources)
    recorder = Recorder(settings.roofline_debug)
    ctx = Context(image, settings, roof_masks, keep, blockers, vp_region=vp_region, crop=crop)
    recorder.counts["sam"] = {"roof_masks": len(roof_masks), "blockers": bool(blockers is not None and blockers.any())}
    recorder.counts["target_house"] = {"planes": int(sum(keep)) if keep is not None else len(roof_masks),
                                       "of": len(roof_masks), "crop": crop.as_list() if crop else None,
                                       "crop_size": [crop.width, crop.height] if crop else None}

    use_v3 = v3 is not None and v3.usable
    report: dict = {"sources": sources, "roof_lines_from": "v3" if use_v3 else "partition"}
    if v3 is not None:
        report.update({"v3_status": v3.status, "fallback_used": not use_v3,
                       "v3_error": v3.error or ("V3 returned no lines" if v3.status == "empty" else None),
                       "crop": v3.crop.as_list() if v3.crop else None, "crop_basis": v3.crop_basis,
                       "v3_cache": v3.meta.get("cache"), "v3": v3.stored(),
                       "v3_missed_planes": v3.meta.get("missed_planes")})

    details: dict = {}
    if use_v3:
        lines = refine_v3(v3, ctx, recorder)
        final, occlusion_counts = handle_occlusion(lines, ctx)
        recorder.counts["occlusion"] = occlusion_counts
        recorder.stage("H_occlusion", final)
        for line in final:
            for piece in line.bridged_from or []:
                recorder.note(piece, occlusion="bridged", bridged_into=line.source_id)
            if line.evidence.get("extended_behind"):
                recorder.note(line.source_id, occlusion="extended")
        requested = [_wanted(line, parsed) for line in final]
        # Lit lines first, then the ones kept for review, as the editor has always listed them.
        order = sorted(range(len(final)), key=lambda i: (not requested[i],))
        final, requested = [final[i] for i in order], [requested[i] for i in order]
    else:
        # Partition: the existing lines, unchanged (occlusion already ran inside), with vanishing-
        # point metadata and confidence added.
        if v3 is not None:
            recorder.counts["v3"] = {"raw_lines": len(v3.raw_lines), "valid_lines": 0}
        spans: list[tuple] = []
        unrequested: list[tuple] = []
        if partition_lines is not None:
            # The side filter is applied below, once sides come from the vanishing points; the
            # partition's own sides are plane-slope labels.
            spans, unrequested, details = partition_lines(dataclasses.replace(parsed, sides=[]))
        inferred_keys = set(details.get("inferred", set()) or set())
        final = [normalize_partition_line(s, i, inferred_keys) for i, s in enumerate(spans + unrequested, 1)]
        for line in final:
            assign_vanishing_point(line, ctx)
        # Sides come from the vanishing points, as V3's do (refine_v3), never from the partition's
        # own plane-slope label: a line with no facade has no side, and is kept for review.
        assign_facade_by_adjacency(final, ctx)
        for line in final:
            line.side = line.facade
        requested = [i < len(spans) and _wanted(line, parsed) for i, line in enumerate(final)]
        order = sorted(range(len(final)), key=lambda i: (not requested[i],))
        final, requested = [final[i] for i in order], [requested[i] for i in order]
        recorder.counts["partition"] = {"lines": len(final)}
        recorder.stage("P_partition", final)

    for line in final:
        calculate_confidence(line, final, ctx)
    confidences = [line.confidence for line in final if line.confidence is not None]
    recorder.counts["final"] = {
        "total": len(final), "lit": int(sum(requested)),
        "high": sum(1 for c in confidences if level(c, settings) == "high"),
        "medium": sum(1 for c in confidences if level(c, settings) == "medium"),
        "low": sum(1 for c in confidences if level(c, settings) == "low"),
    }
    if use_v3:
        for line in final:
            for member in [line.source_id, *(line.bridged_from or [])]:
                recorder.note(member, final=True, final_id=line.source_id, confidence=line.confidence)
        for record in recorder.records.values():
            record.setdefault("final", False)
            if "occlusion" not in record and record.get("final"):
                record["occlusion"] = "unchanged"
    recorder.stage("I_final", final)

    report["counts"] = recorder.counts
    report["confidence_levels"] = {"high": settings.roofline_confidence_high, "low": settings.roofline_confidence_low}
    log.info("roof lines from %s: %s", report["roof_lines_from"], _count_line(recorder.counts))
    final_inferred = {tuple(map(tuple, line.int_points())) for line in final if line.inferred}
    return Result(final, requested, final_inferred, details, report, recorder.export(), ctx.scene)


def _count_line(counts: dict) -> str:
    parts = []
    for name in ("sam", "v3", "partition", "occlusion", "final"):
        if name in counts:
            parts.append(f"{name} " + " ".join(f"{k}={v}" for k, v in counts[name].items()))
    return " | ".join(parts)


def _wanted(line: RoofLine, parsed) -> bool:
    """The instruction's side and edge filter, as roof_lines applies it. A line with no side (V3
    without a vanishing point) cannot answer a side-specific request: kept for review, not lit."""
    edge = line.line_type or "unknown"
    if parsed.sides and line.side not in parsed.sides:
        return False
    return not parsed.edges or edge in parsed.edges or edge == "unknown"


# -- debug overlays ----------------------------------------------------------------------------------------


STAGE_COLOURS = {"v3": (0, 200, 255), "partition": (255, 159, 10)}


def write_overlays(image: Image.Image, debug: dict | None, directory: Path) -> list[str]:
    """One PNG per recorded stage, for comparing stages side by side offline. Development only."""
    if not debug:
        return []
    from PIL import ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, lines in debug["stages"].items():
        canvas = image.copy().convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for line in lines:
            points = line.get("points")
            if not points:
                continue  # A_raw_v3 is in crop pixels; it is compared in the editor, not here
            sources = line.get("source") or ["v3"]
            colour = STAGE_COLOURS.get(sources[0], STAGE_COLOURS["v3"])
            draw.line([tuple(p) for p in points], fill=colour, width=3)
        path = directory / f"roofline_{name}.png"
        canvas.save(path)
        written.append(path.name)
    return written
