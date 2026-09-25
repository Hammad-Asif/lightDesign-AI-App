"""Phase 7: roles from topology, and pitch.

Roles do not come from angle thresholds against the image axes or from a height cutoff (both
depend on the view). They come from what an edge separates:
    shared by two planes, horizontal in plan        -> ridge
    shared by two planes, sloped, convex            -> hip
    shared by two planes, sloped, concave           -> valley   (its lower end sits in a re-entrant
                                                                 corner of the roof outline)
    shared, sloped, the smaller plane sits ON the   -> rake     (a dormer: its free edges are interior
    larger one                                                   to the outline of the two, so it
                                                                 overhangs with a fascia; a valley is flush)
    free, horizontal in plan, the plane's low edge  -> eave     (its high edge is a silhouette ridge)
    free, sloped                                    -> rake, or hip when the plane narrows upward
                                                       (a hip roof's silhouette edges are hips)
"Horizontal in plan" means: points at a horizontal vanishing point. Only when no VP is usable does
it fall back to the image horizontal.

Pitch is reported three ways and always labelled: apparent (the 2-D slope against the local
horizontal, valid only as seen in the photo), calibrated (true dihedral angle, needs a focal
length: EXIF, user input, or the estimate from the vanishing points), and the nearest standard
pitch by a MAP estimate, which is often the number a roofer actually wants.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .config import Params
from .vanishing import VanishingPoints

STANDARD_PITCHES = {2: 0.3, 3: 0.7, 4: 1.6, 5: 1.4, 6: 1.6, 7: 1.0, 8: 1.2, 9: 0.8, 10: 0.8, 12: 1.0, 14: 0.3, 16: 0.2}   # rise per 12 : prior
ROLE_COLORS = {"ridge": "#ff3b30", "hip": "#ff9f0a", "valley": "#0a84ff", "eave": "#39ff14", "rake": "#bf5af2", "inferred": "#8e8e93"}


def _horizontal_dirs(vps: Optional[VanishingPoints], point: np.ndarray) -> list[np.ndarray]:
    if vps is None:
        return []
    return [d for d in (vps.direction_at(h, point) for name, h in vps.named() if name.startswith("h") and vps.supports.get(name, 0) >= 0.10) if d is not None]


def _is_horizontal(u: np.ndarray, mid: np.ndarray, vps: Optional[VanishingPoints], params: Params) -> bool:
    dirs = _horizontal_dirs(vps, mid)
    if dirs:
        return any(abs(float(u @ d)) >= math.cos(math.radians(params.horizontal_vp_deg)) for d in dirs)
    return abs(math.degrees(math.atan2(u[1], u[0])) % 180 - 90) > 78           # no usable VP: image horizontal


def _local_horizontal(u: np.ndarray, mid: np.ndarray, vps: Optional[VanishingPoints]) -> np.ndarray:
    """The horizontal direction an edge's slope should be measured against: the facade's, not the image's."""
    dirs = _horizontal_dirs(vps, mid)
    if not dirs:
        return np.array([1.0, 0.0])
    return min(dirs, key=lambda d: abs(d[1]))                                   # the one closest to the image horizontal


def standard_pitch(angle_deg: float, sigma_deg: float) -> dict:
    best, total = None, 0.0
    scores = {}
    for rise, prior in STANDARD_PITCHES.items():
        a = math.degrees(math.atan(rise / 12.0))
        scores[rise] = prior * math.exp(-0.5 * ((angle_deg - a) / sigma_deg) ** 2)
        total += scores[rise]
    if total <= 0:
        return {"pitch": None, "probability": 0.0}
    best = max(scores, key=scores.get)
    return {"pitch": f"{best}:12", "probability": round(scores[best] / total, 3)}


def assign(planes: list[dict], segments: list[dict], labels: np.ndarray, vps: Optional[VanishingPoints], params: Params) -> None:
    H, W = labels.shape
    radius = max(6, int(0.02 * max(H, W)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)) > 0

    def occupancy(x: int, y: int) -> float:
        """Share of a disc around (x, y) that is roof: under a half at a convex corner, over it in a re-entrant one."""
        x0, x1, y0, y1 = max(x - radius, 0), min(x + radius + 1, W), max(y - radius, 0), min(y + radius + 1, H)
        window = disc[y0 - (y - radius):y1 - (y - radius), x0 - (x - radius):x1 - (x - radius)]
        return float(((labels[y0:y1, x0:x1] > 0) & window).sum()) / float(disc.sum())

    polygons: dict[int, list[np.ndarray]] = {}                                    # a plane an object splits has parts
    for pl in planes:
        polygons.setdefault(pl["plane"], []).append(np.array(pl["polygon"], np.float32).reshape(-1, 1, 2))
    enclosure_cache: dict[tuple[int, int], bool] = {}

    def enclosed(child: int, host: int) -> bool:
        """A dormer sits ON its host: the free edges of the smaller plane (its eave over the dormer face)
        are interior to the outline of host + child, not on it. A wing or a nested hip roof has its free
        edges on the outer outline. Only a dormer's creases are rakes with a fascia; the rest are valleys."""
        key = (child, host)
        if key not in enclosure_cache:
            union = ((labels == child) | (labels == host)).astype(np.uint8)
            contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outer = np.zeros_like(union)
            cv2.drawContours(outer, contours, -1, 1, thickness=5)                 # the outline of host + child, 5 px wide
            region = (labels == child).astype(np.uint8)
            others = np.where(labels == child, 0, labels).astype(np.uint16)
            near_other = cv2.dilate(others, np.ones((7, 7), np.uint8)) > 0       # same reach as the boundary tags
            cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            free = [pt for c in cs for pt in c.reshape(-1, 2) if not near_other[pt[1], pt[0]]]
            if len(free) < 8:
                enclosure_cache[key] = False
            else:
                interior = sum(1 for x, y in free if not outer[y, x])
                enclosure_cache[key] = interior >= 0.8 * len(free)
        return enclosure_cache[key]

    areas = {int(k): int(v) for k, v in zip(*np.unique(labels[labels > 0], return_counts=True))}

    geometry = []
    for s in segments:
        p1, p2 = np.array(s["points"][0], float), np.array(s["points"][1], float)
        d = p2 - p1
        length = float(np.linalg.norm(d))
        u = d / length if length > 1e-9 else np.array([1.0, 0.0])
        mid = (p1 + p2) / 2
        geometry.append((p1, p2, u, mid, length, _is_horizontal(u, mid, vps, params)))

    # per plane: its horizontal free edges, to tell eave from silhouette ridge and hip roofs from gables
    by_plane: dict[int, list[int]] = {}
    for i, s in enumerate(segments):
        for p in s.get("planes", []):
            by_plane.setdefault(p, []).append(i)
    narrows_upward: dict[int, bool] = {}
    for p, idx in by_plane.items():
        horizontals = [i for i in idx if geometry[i][5]]
        if not horizontals:
            narrows_upward[p] = False
            continue
        low = max(horizontals, key=lambda i: geometry[i][3][1])
        tops = [i for i in horizontals if geometry[i][3][1] < geometry[low][3][1] - 0.25 * radius]
        top_len = max((geometry[i][4] for i in tops), default=0.0)               # an apex counts as a zero-length top
        # Perspective alone shortens a gable plane's far (top) edge to about 0.8 of its eave; a hip roof's
        # ridge is far shorter than that, so the split is drawn well below the perspective effect.
        narrows_upward[p] = top_len < 0.70 * geometry[low][4]

    for i, s in enumerate(segments):
        p1, p2, u, mid, length, horizontal = geometry[i]
        shared = len(s.get("planes", [])) >= 2
        if shared:
            if horizontal:
                role = "ridge"
            else:
                a, b = s["planes"][0], s["planes"][1]
                child, host = (a, b) if areas.get(a, 0) <= areas.get(b, 0) else (b, a)
                low_end = p1 if p1[1] > p2[1] else p2
                x, y = int(np.clip(round(low_end[0]), 0, W - 1)), int(np.clip(round(low_end[1]), 0, H - 1))
                if enclosed(child, host):
                    role = "rake"                                                  # a dormer's roof edge over the host
                else:
                    role = "valley" if occupancy(x, y) > 0.58 else "hip"
        elif horizontal:
            # a roof plane rises from its eave: if the plane lies above this edge it is an eave, if below a ridge
            plane = s["planes"][0] if s.get("planes") else None
            parts = polygons.get(plane, [])
            normal = np.array([-u[1], u[0]])
            if normal[1] > 0:
                normal = -normal                                                # pointing up in the image
            step = max(3.0, 0.004 * max(H, W))
            above = any(cv2.pointPolygonTest(poly, tuple(map(float, mid + step * normal)), False) > 0 for poly in parts)
            below = any(cv2.pointPolygonTest(poly, tuple(map(float, mid - step * normal)), False) > 0 for poly in parts)
            if parts and not above and not below:
                # a plane too thin for the probe (a slope seen nearly edge-on): the centroid of its nearest part
                # says which side of the edge it is on
                nearest = min(parts, key=lambda poly: abs(cv2.pointPolygonTest(poly, tuple(map(float, mid)), True)))
                centroid = nearest.reshape(-1, 2).mean(axis=0)
                below = float((centroid - mid) @ normal) < 0
            role = "ridge" if (below and not above) else "eave"
        else:
            plane = s["planes"][0] if s.get("planes") else None
            role = "hip" if narrows_upward.get(plane, False) else "rake"
        s["role"] = role

        # apparent slope, measured against the facade's horizontal at this edge (robust to camera roll)
        h = _local_horizontal(u, mid, vps)
        run, rise = float(u @ h), float(u @ np.array([-h[1], h[0]]))
        if run < 0:
            run, rise = -run, -rise
        angle = math.degrees(math.atan2(-rise, run)) if abs(run) > 1e-9 else 90.0
        s["length_px"] = round(length, 2)
        s["angle_deg"] = round(angle, 2) + 0.0
        s["slope"] = None if abs(angle) >= 89.9 else round(math.tan(math.radians(angle)), 5) + 0.0
        s["slope_pixel"] = None if abs(p2[0] - p1[0]) < 1e-9 else round(float((p2[1] - p1[1]) / (p2[0] - p1[0])), 5) + 0.0
        s["pitch_apparent"] = f"{abs(math.tan(math.radians(angle))) * 12:.1f}:12" if abs(angle) <= 75 else None
        s["color"] = ROLE_COLORS["inferred"] if not s.get("fitted", True) else ROLE_COLORS[role]


def calibrated_pitch(planes: list[dict], segments: list[dict], vps: Optional[VanishingPoints], focal_px: Optional[float],
                     shape: tuple[int, int], params: Params) -> dict:
    """True pitch per plane from a focal length, the vertical VP and the plane's eave VP.

    A rake lies in the vertical plane perpendicular to its eave, so its 3-D direction is
    cos(t) * across + sin(t) * up, and that direction's vanishing point must lie on the rake's image
    line: one equation, one unknown t (the pitch). A hip of an equal-pitch roof runs at 45 degrees in
    plan, which gives the same kind of equation for the hip's own slope p, and tan(t) = sqrt(2) tan(p).
    """
    out = {"focal_px": focal_px, "planes": {}}
    if vps is None or focal_px is None or vps.vertical is None or not vps.horizontals:
        return out
    H, W = shape
    K = np.array([[focal_px, 0, W / 2.0], [0, focal_px, H / 2.0], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)

    def to_pixel_h(vp: np.ndarray) -> np.ndarray:                                # centred/scaled homogeneous -> pixel homogeneous
        return np.array([vp[0] * vps.scale + vp[2] * vps.centre[0], vp[1] * vps.scale + vp[2] * vps.centre[1], vp[2]])

    up = Kinv @ to_pixel_h(vps.vertical)
    up /= np.linalg.norm(up)
    if up[1] > 0:
        up = -up                                                                 # image y points down
    for plane in planes:
        estimates = []
        mine = [s for s in segments if plane["plane"] in s.get("planes", [])]
        eaves = [s for s in mine if s.get("role") in ("eave", "ridge") and s.get("vp", "") and str(s["vp"]).startswith("h")]
        if not eaves:
            continue
        name = max(eaves, key=lambda s: s["length_px"])["vp"]
        along = Kinv @ to_pixel_h(dict(vps.named())[name])
        along -= (along @ up) * up                                               # exactly horizontal
        along /= np.linalg.norm(along)
        across = np.cross(up, along)
        for s in mine:
            if s.get("role") not in ("rake", "hip", "valley") or not s.get("fitted", True):
                continue
            a, b = np.array([*s["points"][0], 1.0]), np.array([*s["points"][1], 1.0])
            line = np.cross(a, b)
            lk = line @ K
            denom = float(lk @ up)
            if abs(denom) < 1e-12:
                continue
            if s["role"] == "rake":
                estimates.append((abs(math.degrees(math.atan(-float(lk @ across) / denom))), s["length_px"]))
            else:
                options = [abs(math.degrees(math.atan(math.sqrt(2) * (-float(lk @ ((along * sgn + across) / math.sqrt(2))) / denom)))) for sgn in (1, -1)]
                plausible = [o for o in options if 8 <= o <= 65]
                if plausible:
                    estimates.append((min(plausible, key=lambda o: abs(o - 30)) if len(plausible) > 1 else plausible[0], s["length_px"]))
        estimates = [(v, w) for v, w in estimates if 5 <= v <= 70]
        if len(estimates) >= 2 and max(v for v, _ in estimates) - min(v for v, _ in estimates) > 10.0:
            continue            # the plane's own edges disagree: its eave direction was probably misread, so say nothing
        if estimates:
            values, weights = np.array([v for v, _ in estimates]), np.array([w for _, w in estimates])
            order = np.argsort(values)
            median = float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])
            out["planes"][plane["plane"]] = {"pitch_deg": round(median, 2), "pitch": f"{math.tan(math.radians(median)) * 12:.1f}:12",
                                             "standard": standard_pitch(median, params.pitch_sigma_calibrated_deg), "edges_used": len(estimates),
                                             "weight": float(weights.sum())}
    if out["planes"]:                                                            # one roof usually has one pitch
        values = np.array([v["pitch_deg"] for v in out["planes"].values()])
        weights = np.array([v.pop("weight") for v in out["planes"].values()])
        order = np.argsort(values)
        roof = float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])
        out["roof"] = {"pitch_deg": round(roof, 2), "pitch": f"{math.tan(math.radians(roof)) * 12:.1f}:12",
                       "standard": standard_pitch(roof, params.pitch_sigma_calibrated_deg)}
    return out
