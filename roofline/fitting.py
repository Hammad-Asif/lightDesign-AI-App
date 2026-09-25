"""Phase 3: per-plane line fitting.

approxPolyDP only SEEDS corners. The geometry that comes out is made of lines fitted by Huber
total least squares to the trusted boundary pixels, and of corners that are intersections of
those lines. Where a tree or the frame hides part of the boundary, the edges there are not fitted
at all: the neighbouring lines are extended through the gap and intersected, which reproduces a
human drawing the roof edge straight through the tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .cleanup import Boundary
from .config import Params


# --------------------------------------------------------------------------------------------
# Small geometry helpers, shared with the later phases
# --------------------------------------------------------------------------------------------

def moments_of(points: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """[S0, Sx, Sy, Sxx, Sxy, Syy]: enough to refit a line in O(1) and to add evidence later."""
    w = np.ones(len(points)) if weights is None else weights
    x, y = points[:, 0], points[:, 1]
    return np.array([w.sum(), (w * x).sum(), (w * y).sum(), (w * x * x).sum(), (w * x * y).sum(), (w * y * y).sum()])


def line_from_moments(m: np.ndarray, prior_dir: Optional[np.ndarray] = None, prior_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line from moments. An optional direction prior adds the quadratic
    penalty prior_weight * sin^2(angle to prior_dir), which keeps the problem a 2x2 eigenproblem."""
    s0 = max(m[0], 1e-9)
    mean = np.array([m[1], m[2]]) / s0
    cov = np.array([[m[3] - s0 * mean[0] ** 2, m[4] - s0 * mean[0] * mean[1]],
                    [m[4] - s0 * mean[0] * mean[1], m[5] - s0 * mean[1] ** 2]])
    if prior_dir is not None and prior_weight > 0:
        # the fit minimises n^T cov n over unit normals n; penalise the normal's component along the prior
        cov = cov + prior_weight * np.outer(prior_dir, prior_dir)
    values, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]                       # smallest eigenvalue
    return mean, np.array([-normal[1], normal[0]])


def intersect(o1, d1, o2, d2) -> Optional[np.ndarray]:
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((o2[0] - o1[0]) * d2[1] - (o2[1] - o1[1]) * d2[0]) / cross
    return o1 + t * d1


def project(point, origin, direction) -> np.ndarray:
    return origin + float((point - origin) @ direction) * direction


def angle_between(d1, d2) -> float:
    """Directed angle in degrees, 0..180."""
    return math.degrees(math.acos(float(np.clip(d1 @ d2, -1.0, 1.0))))


def undirected_angle(d1, d2) -> float:
    return math.degrees(math.acos(float(np.clip(abs(d1 @ d2), 0.0, 1.0))))


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

@dataclass
class FitEdge:
    plane: int
    part: int
    origin: np.ndarray
    direction: np.ndarray                 # unit, oriented along the boundary traversal
    moments: np.ndarray
    first_px: np.ndarray                  # first / last supporting boundary pixel
    last_px: np.ndarray
    tags: dict[int, int]                  # boundary pixels per neighbour (0 = free)
    gap_before: float = 0.0               # unsupported boundary length between the previous edge and this one
    bridged: float = 0.0                  # occluded length this edge was extended across
    fitted: bool = True                   # False: connector across a gap that could not be recovered
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None
    start_inferred: bool = False          # corner recovered through an occlusion gap
    end_inferred: bool = False
    vp: Optional[str] = None              # vanishing point this edge was pulled to (phase 4)
    direction_group: int = -1             # shared-2D-direction cluster (phase 4), -1 if none
    prior_dir: Optional[np.ndarray] = None
    prior_weight: float = 0.0
    cluster: int = -1                     # infinite-line cluster in the planar graph (phase 5)

    @property
    def support(self) -> float:
        return float(self.moments[0])

    @property
    def span(self) -> float:
        return float(np.linalg.norm(self.last_px - self.first_px))

    @property
    def stiffness(self) -> float:
        """How hard the pixels resist a rotation of the line about its centroid: N * L^2 / 12."""
        return self.support * self.span ** 2 / 12.0

    @property
    def centroid(self) -> np.ndarray:
        return np.array([self.moments[1], self.moments[2]]) / max(self.moments[0], 1e-9)

    def neighbour(self) -> int:
        """The plane on the other side of most of this edge, or 0 if it is mostly free."""
        others = {k: v for k, v in self.tags.items() if k != 0}
        if not others or self.tags.get(0, 0) >= 0.5 * sum(self.tags.values()):
            return 0
        return max(others, key=others.get)


@dataclass
class PlanePolygon:
    plane: int
    part: int
    fitted: list[FitEdge] = field(default_factory=list)   # the fitted lines, in boundary order
    edges: list[FitEdge] = field(default_factory=list)    # closed cycle: fitted lines plus any connectors
    eps_px: float = 1.0

    def vertices(self) -> np.ndarray:
        return np.array([e.start for e in self.edges])


# --------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------

def _seed_indices(boundary: Boundary, eps_px: float) -> list[int]:
    contour = boundary.contour
    approx = cv2.approxPolyDP(contour.reshape(-1, 1, 2), eps_px, True).reshape(-1, 2)
    lookup: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(contour):
        lookup.setdefault((int(x), int(y)), []).append(i)
    seeds, previous = [], -1
    for x, y in approx:
        candidates = lookup.get((int(x), int(y)), [])
        if not candidates:
            continue
        ahead = [c for c in candidates if c > previous]
        index = ahead[0] if ahead else candidates[0]
        seeds.append(index)
        previous = index
    # a change of neighbour along the boundary is a likely corner too (three regions meet there)
    n = len(contour)
    changes = [int(i) for i in np.nonzero(boundary.tags != np.roll(boundary.tags, 1))[0]]
    min_sep = max(3.0, eps_px / 2.0)
    for c in changes:
        if all(min((c - s) % n, (s - c) % n) >= min_sep for s in seeds):
            seeds.append(c)
    return sorted(set(seeds))


def _fit_run(points: np.ndarray, trusted: np.ndarray, params: Params):
    use = points[trusted]
    if len(use) < max(params.min_trusted_px, params.min_trusted_frac * len(points)):
        return None
    vx, vy, x0, y0 = cv2.fitLine(use.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    direction = np.array([vx, vy], float)
    origin = np.array([x0, y0], float)
    residual = np.abs((use - origin) @ np.array([-direction[1], direction[0]]))
    inliers = use[residual <= params.huber_inlier_px]
    if len(inliers) < params.min_trusted_px:
        inliers = use
    if np.dot(direction, points[-1] - points[0]) < 0:
        direction = -direction
    moments = moments_of(inliers)
    origin, refit_dir = line_from_moments(moments)
    if np.dot(refit_dir, direction) < 0:
        refit_dir = -refit_dir
    return origin, refit_dir, moments, inliers[0], inliers[-1]


def fit_boundary(boundary: Boundary, eps_px: float, params: Params, long_edge: int, respect_trust: bool = True) -> Optional[PlanePolygon]:
    pts = boundary.contour.astype(float)
    n = len(pts)
    seeds = _seed_indices(boundary, eps_px)
    if len(seeds) < 3:
        return None
    trusted = boundary.trusted if respect_trust else np.ones(n, bool)

    edges: list[FitEdge] = []
    pending_gap = 0.0
    for k in range(len(seeds)):
        a, b = seeds[k], seeds[(k + 1) % len(seeds)]
        idx = np.arange(a, b + 1) if b >= a else np.concatenate([np.arange(a, n), np.arange(0, b + 1)])
        fit = _fit_run(pts[idx], trusted[idx], params)
        if fit is None:
            pending_gap += len(idx)
            continue
        origin, direction, moments, first_px, last_px = fit
        tag_ids, tag_counts = np.unique(boundary.tags[idx], return_counts=True)
        edges.append(FitEdge(boundary.plane, boundary.part, origin, direction, moments, first_px, last_px,
                             {int(t): int(c) for t, c in zip(tag_ids, tag_counts)}, gap_before=pending_gap))
        pending_gap = 0.0
    if not edges:
        return fit_boundary(boundary, eps_px, params, long_edge, respect_trust=False) if respect_trust else None
    edges[0].gap_before += pending_gap                        # the boundary is cyclic
    if len(edges) < 2:
        return fit_boundary(boundary, eps_px, params, long_edge, respect_trust=False) if respect_trust else None

    # ---- merge neighbours that are one line: same direction, and collinear even across a gap
    collinear_tol = max(3.0, eps_px)
    while len(edges) > 3:
        best, best_angle = -1, params.merge_angle_deg
        for k in range(len(edges)):
            a, b = edges[k], edges[(k + 1) % len(edges)]
            ang = angle_between(a.direction, b.direction)
            if ang >= best_angle:
                continue
            na = np.array([-a.direction[1], a.direction[0]])
            nb = np.array([-b.direction[1], b.direction[0]])
            if abs(float((b.origin - a.origin) @ na)) > collinear_tol or abs(float((a.origin - b.origin) @ nb)) > collinear_tol:
                continue
            best, best_angle = k, ang
        if best < 0:
            break
        j = (best + 1) % len(edges)
        a, b = edges[best], edges[j]
        moments = a.moments + b.moments
        origin, direction = line_from_moments(moments)
        if np.dot(direction, a.direction) < 0:
            direction = -direction
        tags = dict(a.tags)
        for t, c in b.tags.items():
            tags[t] = tags.get(t, 0) + c
        edges[best] = FitEdge(a.plane, a.part, origin, direction, moments, a.first_px, b.last_px, tags,
                              gap_before=a.gap_before, bridged=a.bridged + b.bridged + b.gap_before)
        del edges[j]

    polygon = PlanePolygon(boundary.plane, boundary.part, fitted=edges, eps_px=eps_px)
    return polygon if close_polygon(polygon, params, long_edge) else None


def close_polygon(polygon: PlanePolygon, params: Params, long_edge: int) -> bool:
    """(Re)compute every corner as the intersection of consecutive fitted lines."""
    edges, eps_px = polygon.fitted, polygon.eps_px
    for e in edges:
        e.start = e.end = None
        e.start_inferred = e.end_inferred = False
    guard = max(2.0 * eps_px, params.corner_guard_frac * long_edge)
    out: list[FitEdge] = []
    for k in range(len(edges)):
        a, b = edges[k - 1], edges[k]
        p = intersect(a.origin, a.direction, b.origin, b.direction)
        if b.gap_before <= 0:
            seed = b.first_px
            if p is None or np.linalg.norm(p - seed) > guard:
                p = (project(seed, a.origin, a.direction) + project(seed, b.origin, b.direction)) / 2.0
            a.end, b.start = p, p
            continue
        # a gap: the boundary between a and b is hidden. Extend both lines through it.
        sa, sb = a.last_px, b.first_px
        gap_len = float(np.linalg.norm(sa - sb))
        ok = p is not None
        if ok:
            ahead_of_a = float((p - sa) @ a.direction) >= -guard
            before_b = float((sb - p) @ b.direction) >= -guard
            near = max(np.linalg.norm(p - sa), np.linalg.norm(p - sb)) <= params.gap_guard_ratio * gap_len + guard
            ok = ahead_of_a and before_b and near
        if ok:
            a.end, b.start = p, p
            a.end_inferred = b.start_inferred = True
        else:                                                  # cannot be recovered: bridge with a plain connector
            a.end = project(sa, a.origin, a.direction)
            b.start = project(sb, b.origin, b.direction)
            d = b.start - a.end
            norm = np.linalg.norm(d)
            if norm > 1.0:
                connector = FitEdge(a.plane, a.part, a.end.copy(), d / norm, moments_of(np.array([a.end, b.start])), a.end, b.start,
                                    {}, fitted=False, start=a.end.copy(), end=b.start.copy(), start_inferred=True, end_inferred=True)
                out.append(("before", k, connector))
    connectors = {k: c for _, k, c in out}
    polygon.edges = []
    for k, e in enumerate(edges):
        if k in connectors:
            polygon.edges.append(connectors[k])
        polygon.edges.append(e)
    # the connector "before edge 0" closes the cycle, so it belongs at the end
    if 0 in connectors:
        polygon.edges.append(polygon.edges.pop(0))
    return len(polygon.edges) >= 3


def fit_planes(boundaries: list[Boundary], params: Params, long_edge: int) -> list[PlanePolygon]:
    if not boundaries:
        return []
    reference = max(b.perimeter for b in boundaries)
    polygons = []
    for b in boundaries:
        eps_px = min(params.epsilon * reference, params.eps_cap_own_frac * b.perimeter)
        polygon = fit_boundary(b, max(eps_px, 1.0), params, long_edge)
        if polygon is not None:
            polygons.append(polygon)
    return polygons


# --------------------------------------------------------------------------------------------
# Output for inspection (phases 3 and 4 have no graph yet, so shared borders are merged here)
# --------------------------------------------------------------------------------------------

def polygons_to_segments(polygons: list[PlanePolygon], merge_tol_px: float, merge_angle_deg: float) -> list[dict]:
    raw = []
    for poly in polygons:
        for e in poly.edges:
            if e.start is None or e.end is None or np.linalg.norm(e.end - e.start) < 1.0:
                continue
            raw.append({"p1": e.start.copy(), "p2": e.end.copy(), "plane": poly.plane, "neighbour": e.neighbour(),
                        "inferred": bool(e.start_inferred or e.end_inferred or e.bridged > 0), "fitted": e.fitted, "planes": [poly.plane]})
    cos_limit = math.cos(math.radians(merge_angle_deg))
    kept: list[dict] = []
    for seg in sorted(raw, key=lambda s: -np.linalg.norm(s["p2"] - s["p1"])):
        merged = False
        for base in kept:
            if seg["plane"] in base["planes"]:
                continue
            d = base["p2"] - base["p1"]
            length = float(np.linalg.norm(d))
            d = d / length
            normal = np.array([-d[1], d[0]])
            sd = seg["p2"] - seg["p1"]
            slen = float(np.linalg.norm(sd))
            if abs(float(d @ (sd / slen))) < cos_limit:
                continue
            if abs(float((seg["p1"] - base["p1"]) @ normal)) > merge_tol_px or abs(float((seg["p2"] - base["p1"]) @ normal)) > merge_tol_px:
                continue
            t1, t2 = sorted((float((seg["p1"] - base["p1"]) @ d), float((seg["p2"] - base["p1"]) @ d)))
            if min(length, t2) - max(0.0, t1) < 0.3 * min(length, slen):
                continue
            lo, hi = min(0.0, t1), max(length, t2)
            base["p1"], base["p2"] = base["p1"] + lo * d, base["p1"] + hi * d
            base["planes"].append(seg["plane"])
            merged = True
            break
        if not merged:
            kept.append(seg)
    return [{"points": [[round(float(s["p1"][0]), 2), round(float(s["p1"][1]), 2)], [round(float(s["p2"][0]), 2), round(float(s["p2"][1]), 2)]],
             "planes": sorted(s["planes"]), "inferred": s["inferred"], "fitted": s["fitted"]} for s in kept]
