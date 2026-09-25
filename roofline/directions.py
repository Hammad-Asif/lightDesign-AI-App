"""Phase 4b: direction regularisation.

Roofs have very few real directions. Eaves, ridges and tile rows run along the scene's horizontal
directions, so an edge whose fitted direction is close to a vanishing-point direction is pulled
onto it. Hips and rakes do not point at a Manhattan VP; they are regularised in 2-D instead: edges
that nearly share a direction are made to share it exactly (this is also the whole fallback when
the VPs are unstable), and opposite rakes that are nearly mirror images are made symmetric.

The pull is a quadratic prior inside the total-least-squares fit (see line_from_moments), scaled
by each edge's own stiffness, so a long well-supported edge moves little and a short one a lot.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import Params
from .fitting import PlanePolygon, line_from_moments, undirected_angle
from .vanishing import VanishingPoints

MIN_VP_SUPPORT = 0.10          # a VP backed by less than this share of all line length is not trusted


def _axis(direction: np.ndarray) -> float:
    return math.degrees(math.atan2(direction[1], direction[0])) % 180.0


def assign_priors(polygons: list[PlanePolygon], vps: Optional[VanishingPoints], params: Params) -> dict:
    edges = [e for poly in polygons for e in poly.fitted]
    for e in edges:
        e.vp, e.direction_group, e.prior_dir = None, -1, None
    usable = [] if vps is None or params.w_dir <= 0 else [(name, vp) for name, vp in vps.named() if vps.supports.get(name, 0.0) >= MIN_VP_SUPPORT]

    for e in edges:
        best = (params.vp_snap_deg, None, None)
        for name, vp in usable:
            d = vps.direction_at(vp, e.centroid)
            if d is None:
                continue
            ang = undirected_angle(e.direction, d)
            if ang < best[0]:
                best = (ang, name, d)
        if best[1] is not None:
            e.vp, e.prior_dir = best[1], best[2] if best[2] @ e.direction >= 0 else -best[2]

    # everything else: shared 2-D directions. Heaviest edges found clusters; lighter ones join them.
    # Sharing 2-D directions is wrong under perspective (parallel lines converge), and the harness shows it
    # costs accuracy, so it only runs as the fallback when no vanishing point can be trusted.
    cluster_weight = params.w_par if params.w_par > 0 else (params.w_par_fallback if not usable else 0.0)
    free = sorted([e for e in edges if e.vp is None], key=lambda e: -e.stiffness) if cluster_weight > 0 else []
    clusters: list[dict] = []
    for e in free:
        ang = _axis(e.direction)
        home = None
        for c in clusters:
            diff = abs(ang - c["angle"]) % 180.0
            if min(diff, 180.0 - diff) <= params.dir_cluster_deg:
                home = c
                break
        if home is None:
            clusters.append({"angle": ang, "members": [e]})
        else:
            home["members"].append(e)
    groups = 0
    for c in clusters:
        if len(c["members"]) < 2:
            continue
        # weighted circular mean on the doubled angle (directions are axes, not vectors)
        w = np.array([m.stiffness for m in c["members"]])
        a = np.radians([2 * _axis(m.direction) for m in c["members"]])
        mean = 0.5 * math.atan2(float((w * np.sin(a)).sum()), float((w * np.cos(a)).sum()))
        d = np.array([math.cos(mean), math.sin(mean)])
        for m in c["members"]:
            m.direction_group = groups
            m.prior_dir = d if d @ m.direction >= 0 else -d
        groups += 1

    # mirror symmetry of opposite rakes within one plane (meaningful when the facade is seen head-on)
    symmetric = 0
    for poly in (polygons if params.symmetry_deg > 0 else []):
        sloped = [e for e in poly.fitted if e.vp is None and 12.0 < _axis(e.direction) < 168.0 and abs(_axis(e.direction) - 90.0) > 12.0]
        for i in range(len(sloped)):
            for j in range(i + 1, len(sloped)):
                a, b = _axis(sloped[i].direction), _axis(sloped[j].direction)
                if abs((a + b) - 180.0) <= params.symmetry_deg and abs(a - b) > 20.0:
                    wi, wj = sloped[i].stiffness, sloped[j].stiffness
                    half = ((a * wi + (180.0 - b) * wj) / (wi + wj))           # a and 180-b are the same slope
                    for e, ang in ((sloped[i], half), (sloped[j], 180.0 - half)):
                        d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
                        e.prior_dir = d if d @ e.direction >= 0 else -d
                    symmetric += 1
    for e in edges:
        e.prior_weight = 0.0 if e.prior_dir is None else (params.w_dir if e.vp else max(cluster_weight, params.w_par_fallback)) * e.stiffness
    return {"vp_edges": sum(1 for e in edges if e.vp), "direction_groups": groups, "symmetric_pairs": symmetric,
            "usable_vps": [name for name, _ in usable], "fallback_clustering": bool(cluster_weight > 0 and not usable)}


def apply_priors(polygons: list[PlanePolygon], params: Params) -> None:
    """Refit every edge with its direction prior. Lines keep passing through their pixel centroid."""
    for poly in polygons:
        for e in poly.fitted:
            if e.prior_dir is None:
                continue
            origin, direction = line_from_moments(e.moments, e.prior_dir, e.prior_weight)
            e.origin, e.direction = origin, direction if direction @ e.direction >= 0 else -direction
