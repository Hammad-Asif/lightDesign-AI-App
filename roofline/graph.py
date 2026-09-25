"""Phase 5: one planar graph for the whole roof, and the consistency loop that relaxes it.

Per-plane polygons disagree with each other: a shared hip is fitted twice, corners that should
coincide sit a few pixels apart, three lines that should meet at a ridge end form a small triangle.
Here every fitted edge is assigned to an infinite-line cluster (angle + offset), every corner to a
junction (snap radius), and then an energy is minimised by block coordinate descent:

    E = sum over lines   of  fit        (squared distance of the trusted boundary pixels to the line)
      + sum over lines   of  direction  (stiffness-scaled sin^2 of the angle to the VP / shared direction)
      + sum over (junction, line) of  w_junc * squared distance of the junction to each line through it
      + w_cov * coverage (area where the polygons and the partition disagree; reported, see below)

Both blocks have closed forms. Lines step: each line is refit by total least squares to its pixels
PLUS its junctions (as weighted points) PLUS its direction prior, a 2x2 eigenproblem. Junctions
step: each junction moves to the least-squares intersection of its lines, a 2x2 solve. Each step
is the exact minimiser of E for its block, so E never increases; the loop stops when the relative
drop in a round is below `energy_tol`. Direction targets are re-read from the vanishing points at
the lines' new centroids between rounds. Coverage is not differentiable in these variables: it is
evaluated before and after, reported, and used by the outer tuning loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import Params
from .fitting import FitEdge, PlanePolygon, intersect, line_from_moments, moments_of, undirected_angle
from .vanishing import VanishingPoints


@dataclass
class Line:
    id: int
    members: list[FitEdge]
    moments: np.ndarray
    origin: np.ndarray
    direction: np.ndarray
    prior_dir: Optional[np.ndarray] = None
    prior_weight: float = 0.0
    vp: Optional[str] = None

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def distance(self, point: np.ndarray) -> float:
        return float((point - self.origin) @ self.normal)

    def fit_energy(self) -> float:
        m, n = self.moments, self.normal
        rho = float(n @ self.origin)
        return float(n @ np.array([[m[3], m[4]], [m[4], m[5]]]) @ n - 2 * rho * (n @ np.array([m[1], m[2]])) + rho * rho * m[0])

    def direction_energy(self) -> float:
        return 0.0 if self.prior_dir is None else self.prior_weight * float(self.normal @ self.prior_dir) ** 2


@dataclass
class Junction:
    id: int
    position: np.ndarray
    corners: list[tuple[int, int]]                 # (polygon index, corner index)
    lines: set[int] = field(default_factory=set)
    inferred: bool = False


@dataclass
class Graph:
    polygons: list[PlanePolygon]
    lines: list[Line]
    junctions: list[Junction]
    corner_junction: dict[tuple[int, int], int]
    energy_trace: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _interval(edge: FitEdge, origin: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    a, b = float((edge.first_px - origin) @ direction), float((edge.last_px - origin) @ direction)
    return (a, b) if a <= b else (b, a)


def build(polygons: list[PlanePolygon], params: Params, long_edge: int) -> Graph:
    fitted = [(pi, e) for pi, poly in enumerate(polygons) for e in poly.fitted]
    offset_tol = max(3.0, params.line_offset_frac * long_edge)
    max_gap = 0.25 * long_edge

    # ---- infinite-line clusters
    # Two sides of a plane thinner than the offset tolerance (a dormer roof seen nearly edge-on)
    # merge into one line here; that is the tolerance's own definition of "one line", and the
    # polygon then collapses onto it. coalesce() below folds the coincident junctions that leaves
    # together so the collapsed sides are emitted once, not twice.
    uf = _UnionFind(len(fitted))
    for a in range(len(fitted)):
        ea = fitted[a][1]
        na = np.array([-ea.direction[1], ea.direction[0]])
        for b in range(a + 1, len(fitted)):
            eb = fitted[b][1]
            if undirected_angle(ea.direction, eb.direction) >= params.merge_angle_deg:
                continue
            nb = np.array([-eb.direction[1], eb.direction[0]])
            if abs(float((eb.centroid - ea.origin) @ na)) > offset_tol or abs(float((ea.centroid - eb.origin) @ nb)) > offset_tol:
                continue
            ia, ib = _interval(ea, ea.origin, ea.direction), _interval(eb, ea.origin, ea.direction)
            if max(ia[0], ib[0]) - min(ia[1], ib[1]) > max_gap:
                continue
            uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(len(fitted)):
        groups.setdefault(uf.find(i), []).append(i)
    lines: list[Line] = []
    for members in groups.values():
        edges = [fitted[i][1] for i in members]
        moments = np.sum([e.moments for e in edges], axis=0)
        lead = max(edges, key=lambda e: e.stiffness)
        prior_dir = lead.prior_dir
        weight = float(sum(e.prior_weight for e in edges if e.prior_dir is not None)) if prior_dir is not None else 0.0
        origin, direction = line_from_moments(moments, prior_dir, weight)
        if direction @ lead.direction < 0:
            direction = -direction
        line = Line(len(lines), edges, moments, origin, direction, prior_dir, weight, lead.vp)
        for e in edges:
            e.cluster = line.id
        lines.append(line)

    # ---- junctions: corners within the snap radius, complete-linkage so chains cannot run away
    snap = max(3.0, params.snap_radius_frac * long_edge)
    corners = [((pi, ci), e.start) for pi, poly in enumerate(polygons) for ci, e in enumerate(poly.edges)]
    corner_lines = [{e.cluster for e in (polygons[pi].edges[ci - 1], polygons[pi].edges[ci]) if e.fitted and e.cluster >= 0} for (pi, ci), _ in corners]

    def compatible(i: int, j: int) -> bool:
        """Optionally refuse to merge corners whose lines are nearly parallel (thin bands put two corners close)."""
        if params.junction_min_angle_deg <= 0:
            return True
        for a in corner_lines[i]:
            for b in corner_lines[j]:
                if a != b and undirected_angle(lines[a].direction, lines[b].direction) < params.junction_min_angle_deg:
                    return False
        return True

    clusters: list[list[int]] = []
    order = sorted(range(len(corners)), key=lambda i: (corners[i][1][0], corners[i][1][1]))
    for i in order:
        p = corners[i][1]
        home = None
        for c in clusters:
            if all(np.linalg.norm(p - corners[j][1]) <= 2 * snap and compatible(i, j) for j in c) and any(np.linalg.norm(p - corners[j][1]) <= snap for j in c):
                home = c
                break
        if home is None:
            clusters.append([i])
        else:
            home.append(i)
    junctions: list[Junction] = []
    corner_junction: dict[tuple[int, int], int] = {}
    for c in clusters:
        j = Junction(len(junctions), np.mean([corners[i][1] for i in c], axis=0), [corners[i][0] for i in c])
        for i in c:
            pi, ci = corners[i][0]
            corner_junction[(pi, ci)] = j.id
            poly = polygons[pi]
            for e in (poly.edges[ci - 1], poly.edges[ci]):          # the two edges meeting at this corner
                if e.fitted and e.cluster >= 0:
                    j.lines.add(e.cluster)
            j.inferred |= bool(poly.edges[ci].start_inferred)
        junctions.append(j)

    # ---- T-junctions: a corner that lands on the middle of another plane's edge belongs to that line too
    for j in junctions:
        for line in lines:
            if line.id in j.lines or abs(line.distance(j.position)) > snap:
                continue
            t = float((j.position - line.origin) @ line.direction)
            for e in line.members:
                lo, hi = _interval(e, line.origin, line.direction)
                if lo + snap < t < hi - snap:
                    j.lines.add(line.id)
                    break
    return Graph(polygons, lines, junctions, corner_junction)


def _energy(graph: Graph, params: Params) -> dict:
    fit = sum(l.fit_energy() for l in graph.lines) * params.w_fit
    direction = sum(l.direction_energy() for l in graph.lines)
    junction = params.w_junc * sum(graph.lines[i].distance(j.position) ** 2 for j in graph.junctions for i in j.lines)
    return {"fit": fit, "direction": direction, "junction": junction, "total": fit + direction + junction}


def place_junctions(graph: Graph, params: Params, only: Optional[list[Junction]] = None) -> None:
    """Each junction moves to the least-squares intersection of its lines (the exact minimiser of its block)."""
    for j in graph.junctions if only is None else only:
        ls = [graph.lines[i] for i in j.lines]
        if not ls:
            continue
        if len(ls) == 1:
            j.position = j.position - ls[0].distance(j.position) * ls[0].normal
            continue
        A = sum(np.outer(l.normal, l.normal) for l in ls)
        b = sum(l.normal * float(l.normal @ l.origin) for l in ls)
        mu = params.junction_ridge                             # anchored ridge: keeps near-parallel pairs from flying away
        j.position = np.linalg.solve(A + mu * np.eye(2), b + mu * j.position)


def coalesce(graph: Graph, radius: float, params: Params) -> int:
    """Junctions the relaxation has carried to within `radius` of each other become one junction.

    Corners are clustered before the relaxation, on the fitted polygons, where three corners of one
    meeting point can sit tens of pixels apart; the relaxation then moves each to the intersection
    of its lines, and when those are the same lines the junctions coincide, leaving edges a pixel
    or two long between them (owner photo: three corners 57 px apart met at one point and left a
    2 px "ridge"). Complete linkage, like the corner clustering, so chains cannot run away. The
    merged junction is re-placed on the union of its lines. Returns how many junctions were removed.
    """
    n = len(graph.junctions)
    if n < 2 or radius <= 0:
        return 0
    positions = [j.position for j in graph.junctions]
    clusters: list[list[int]] = []
    for i in sorted(range(n), key=lambda k: (float(positions[k][0]), float(positions[k][1]))):
        home = None
        for c in clusters:
            if all(np.linalg.norm(positions[i] - positions[k]) <= radius for k in c):
                home = c
                break
        if home is None:
            clusters.append([i])
        else:
            home.append(i)
    if len(clusters) == n:
        return 0
    remap: dict[int, int] = {}
    merged: list[Junction] = []
    moved: list[Junction] = []
    for c in clusters:
        parts = [graph.junctions[k] for k in c]
        j = Junction(len(merged), np.mean([p.position for p in parts], axis=0), [corner for p in parts for corner in p.corners],
                     set().union(*(p.lines for p in parts)), any(p.inferred for p in parts))
        for k in c:
            remap[k] = j.id
        merged.append(j)
        if len(c) > 1:
            moved.append(j)
    graph.junctions = merged
    graph.corner_junction = {corner: remap[j] for corner, j in graph.corner_junction.items()}
    place_junctions(graph, params, moved)
    return n - len(merged)


def optimise(graph: Graph, vps: Optional[VanishingPoints], params: Params) -> None:
    place_junctions(graph, params)
    graph.energy_trace = [dict(round=0, **_energy(graph, params))]
    incident: dict[int, list[Junction]] = {l.id: [] for l in graph.lines}
    for j in graph.junctions:
        for i in j.lines:
            incident[i].append(j)
    for rnd in range(1, params.max_rounds + 1):
        for line in graph.lines:
            moments = params.w_fit * line.moments
            pts = [j.position for j in incident[line.id]]
            if pts:
                moments = moments + params.w_junc * moments_of(np.array(pts))
            if line.vp and vps is not None:                         # re-read the VP direction where the line is now
                vp = dict(vps.named()).get(line.vp)
                d = vps.direction_at(vp, line.origin) if vp is not None else None
                if d is not None:
                    line.prior_dir = d
            origin, direction = line_from_moments(moments, line.prior_dir, line.prior_weight)
            line.origin, line.direction = origin, direction if direction @ line.direction >= 0 else -direction
        place_junctions(graph, params)
        e = dict(round=rnd, **_energy(graph, params))
        previous = graph.energy_trace[-1]["total"]
        graph.energy_trace.append(e)
        if previous - e["total"] <= params.energy_tol * max(previous, 1e-9):
            break


def coverage(graph: Graph, labels: np.ndarray, scale: float = 0.25) -> dict:
    """Share of the roof area where the polygons and the partition disagree (gaps plus overlaps)."""
    h, w = int(labels.shape[0] * scale), int(labels.shape[1] * scale)
    small = cv2.resize((labels > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    count = np.zeros((h, w), np.uint8)
    for pi, poly in enumerate(graph.polygons):
        pts = np.array([graph.junctions[graph.corner_junction[(pi, ci)]].position for ci in range(len(poly.edges))]) * scale
        layer = np.zeros((h, w), np.uint8)
        cv2.fillPoly(layer, [np.round(pts).astype(np.int32)], 1)
        count += layer
    roof = max(int(small.sum()), 1)
    gaps = int(((count == 0) & (small > 0)).sum()) / roof
    spill = int(((count > 0) & (small == 0)).sum()) / roof
    overlap = int((count > 1).sum()) / roof
    return {"gaps": round(gaps, 4), "spill": round(spill, 4), "overlap": round(overlap, 4), "total": round(gaps + spill + overlap, 4)}


def segments(graph: Graph, params: Params, long_edge: int) -> list[dict]:
    """Unique graph edges. A border shared by two planes appears once, split at every junction on it."""
    out: dict[tuple[int, int], dict] = {}
    on_line: dict[int, list[tuple[float, int]]] = {l.id: [] for l in graph.lines}
    for j in graph.junctions:
        for i in j.lines:
            l = graph.lines[i]
            on_line[i].append((float((j.position - l.origin) @ l.direction), j.id))
    for v in on_line.values():
        v.sort()
    for pi, poly in enumerate(graph.polygons):
        m = len(poly.edges)
        for ci, e in enumerate(poly.edges):
            ja, jb = graph.corner_junction[(pi, ci)], graph.corner_junction[(pi, (ci + 1) % m)]
            if ja == jb:
                continue
            chain = [ja, jb]
            if e.fitted and e.cluster >= 0:                          # split at junctions lying between the ends
                l = graph.lines[e.cluster]
                ta = float((graph.junctions[ja].position - l.origin) @ l.direction)
                tb = float((graph.junctions[jb].position - l.origin) @ l.direction)
                lo, hi = min(ta, tb), max(ta, tb)
                between = [jid for t, jid in on_line[l.id] if lo < t < hi and jid not in (ja, jb)]
                chain = [ja] + (between if ta <= tb else between[::-1]) + [jb]
            for a, b in zip(chain, chain[1:]):
                key = (min(a, b), max(a, b))
                seg = out.setdefault(key, {"junctions": key, "planes": set(), "line": e.cluster if e.fitted else -1, "fitted": e.fitted,
                                           "inferred": False, "vp": None, "neighbours": set()})
                seg["planes"].add(poly.plane)
                seg["fitted"] |= e.fitted
                seg["inferred"] |= bool(e.start_inferred or e.end_inferred or e.bridged > 0 or not e.fitted)
                if e.fitted and e.cluster >= 0:
                    seg["line"], seg["vp"] = e.cluster, graph.lines[e.cluster].vp
                    if e.neighbour():
                        seg["neighbours"].add(e.neighbour())
    result = []
    min_len = max(1.0, params.min_segment_frac * long_edge)
    for (a, b), seg in out.items():
        p1, p2 = graph.junctions[a].position, graph.junctions[b].position
        if np.linalg.norm(p2 - p1) < min_len:
            continue
        result.append({"points": [[round(float(p1[0]), 2), round(float(p1[1]), 2)], [round(float(p2[0]), 2), round(float(p2[1]), 2)]],
                       "junctions": [a, b], "planes": sorted(seg["planes"] | seg["neighbours"]), "line": seg["line"], "vp": seg["vp"],
                       "fitted": seg["fitted"], "inferred": seg["inferred"]})
    return result


def plane_polygons(graph: Graph) -> list[dict]:
    out = []
    for pi, poly in enumerate(graph.polygons):
        ids = []
        for ci in range(len(poly.edges)):
            j = graph.corner_junction[(pi, ci)]
            if not ids or ids[-1] != j:
                ids.append(j)
        if len(ids) > 1 and ids[0] == ids[-1]:
            ids.pop()
        if len(ids) >= 3:
            out.append({"plane": poly.plane, "part": poly.part, "junctions": ids,
                        "polygon": [[round(float(v), 2) for v in graph.junctions[j].position] for j in ids]})
    return out
