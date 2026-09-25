"""
Helpers for the lighting-design editor. Nothing here detects anything new about the roof: roof lines
are the phased pipeline's own segments (see app.design_detect), only regrouped into one polyline per
connected run of the same edge role so a person can light, hide and edit them as lines.
"""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

EDGE_COLOURS = {"eave": "#39ff14", "rake": "#bf5af2", "hip": "#ff9f0a", "ridge": "#ff3b30", "valley": "#0a84ff"}
JOIN_TOL = 1.0


def mask_to_data_url(mask: np.ndarray) -> str:
    """Binary mask -> RGBA PNG (white where set, transparent elsewhere), ready to draw on a canvas."""
    h, w = mask.shape[:2]
    rgba = np.zeros((h, w, 4), np.uint8)
    on = mask > 0
    rgba[on] = (255, 255, 255, 255)
    ok, buf = cv2.imencode(".png", rgba)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def chain_segments(segments: list[dict]) -> list[dict]:
    """Connected runs of same-role segments -> polylines. A run breaks where three or more meet."""
    out: list[dict] = []
    by_role: dict[str, list[dict]] = {}
    for s in segments:
        by_role.setdefault(s["role"], []).append(s)
    for role, segs in by_role.items():
        nodes: list[np.ndarray] = []

        def node_of(p) -> int:
            p = np.asarray(p, float)
            for i, q in enumerate(nodes):
                if np.hypot(*(p - q)) < JOIN_TOL:
                    return i
            nodes.append(p)
            return len(nodes) - 1

        edges = [(node_of(s["points"][0]), node_of(s["points"][1]), s) for s in segs]
        adj: dict[int, list[int]] = {}
        for ei, (a, b, _) in enumerate(edges):
            adj.setdefault(a, []).append(ei)
            adj.setdefault(b, []).append(ei)
        used: set[int] = set()

        def walk(start: int, first_edge: int) -> tuple[list[int], list[int]]:
            path, eids, cur, e = [start], [], start, first_edge
            while e is not None and e not in used:
                used.add(e)
                eids.append(e)
                a, b, _ = edges[e]
                cur = b if a == cur else a
                path.append(cur)
                nxt = [x for x in adj[cur] if x not in used] if len(adj[cur]) == 2 else []
                e = nxt[0] if nxt else None
            return path, eids

        starts = [n for n, es in adj.items() if len(es) != 2] + list(adj)       # ends and branch points first, then loops
        for n in starts:
            for e in adj[n]:
                if e in used:
                    continue
                path, eids = walk(n, e)
                src = [edges[i][2] for i in eids]
                out.append({"edge": role, "polyline": [[round(float(nodes[i][0]), 2), round(float(nodes[i][1]), 2)] for i in path],
                            "planes": sorted({p for s in src for p in s.get("planes", [])}),
                            "inferred": any(s.get("inferred") for s in src),
                            "length_px": round(sum(float(s["length_px"]) for s in src), 1),
                            "pitch_apparent": next((s.get("pitch_apparent") for s in src if s.get("pitch_apparent")), None),
                            "segment_ids": [s.get("id") for s in src]})
    return out


def polyline_length(points: list[list[float]]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def drop_inner_edges(segments: list[dict], band_px: float, angle_deg: float = 8.0) -> list[dict]:
    """A thin roof band (gable trim, a fascia strip) gives two parallel lines, its top and bottom edge. Lights run
    along the top edge, so of two segments that are parallel, within `band_px` of each other and overlap along
    their length by half or more, the lower one is dropped."""
    def geom(sg):
        a, b = np.array(sg["points"][0], float), np.array(sg["points"][1], float)
        d = b - a
        L = float(np.linalg.norm(d)) or 1.0
        return a, b, d / L, L
    keep = [True] * len(segments)
    cos_max = math.cos(math.radians(angle_deg))
    for i, si in enumerate(segments):
        ai, bi, ui, Li = geom(si)
        for j, sj in enumerate(segments):
            if i == j or not keep[i] or not keep[j]:
                continue
            aj, bj, uj, Lj = geom(sj)
            if abs(float(ui @ uj)) < cos_max:
                continue
            n = np.array([-ui[1], ui[0]])
            offset = abs(float((0.5 * (aj + bj) - ai) @ n))
            if offset > band_px:
                continue
            ta, tb = sorted([float((aj - ai) @ ui), float((bj - ai) @ ui)])
            overlap = max(0.0, min(Li, tb) - max(0.0, ta)) / max(1.0, min(Li, Lj))
            if overlap < 0.5:
                continue
            up = n if n[1] < 0 else -n                                   # the normal pointing up the image
            lower = j if float((0.5 * (aj + bj) - ai) @ up) < 0 else i   # parallel lines: the sign holds all along
            keep[lower] = False
    return [sg for sg, k in zip(segments, keep) if k]


def roofline_only(segments: list[dict], found: np.ndarray, band_px: float) -> list[dict]:
    """Of the lines inside a box, the ones lights run along: the upper outline of the roof SAM found there (no
    found roof just above the line), and the eaves of full roof faces (level, with roof well above them). Lines
    with found roof above them otherwise are inner edges of a thin band (a gable's trim) or run through it."""
    H, W = found.shape
    def share(a, d, up, dist) -> float:
        hits = []
        for t in np.linspace(0.15, 0.85, 9):
            p = a + d * t + up * dist
            x, y = int(round(p[0])), int(round(p[1]))
            hits.append(0 <= x < W and 0 <= y < H and bool(found[y, x]))
        return float(np.mean(hits))
    out = []
    for sg in segments:
        a, b = np.array(sg["points"][0], float), np.array(sg["points"][1], float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1:
            continue
        n = np.array([-d[1], d[0]]) / L
        up = n if n[1] < 0 else -n
        if share(a, d, up, 6) <= 0.4:                                          # top outline
            out.append(sg)
        elif abs(math.degrees(math.atan2(abs(d[1]), abs(d[0])))) <= 20 and share(a, d, up, band_px) >= 0.6:
            out.append(sg)                                                      # the eave of a full roof face
    return out
