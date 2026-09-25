"""Synthetic test scenes with exact ground truth.

A small 3-D house (walls, windows, a gable / hip / cross-gable roof) is rendered through a
pinhole camera with a z-buffer. That gives, for free:
  * the ground-truth roof planes as closed image polygons (what a human would annotate),
  * the true pitch of every plane and the true focal length (to test calibrated pitch),
  * real perspective, with wall and window lines that make vanishing points recoverable.

`sam_like_instances` then degrades the true plane regions the way SAM 3 masks are degraded:
wobbly edges, holes and bites from texture, gaps and overlaps between neighbours, an optional
parent instance covering two planes, an optional near-duplicate, an optional single mask that
spans two faces, and bites where a tree stands in front of the roof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------

@dataclass
class Camera:
    f: float
    width: int
    height: int
    position: np.ndarray
    yaw_deg: float      # 0 looks along +Y, positive turns toward +X
    pitch_deg: float    # positive looks up

    def rotation(self) -> np.ndarray:
        yaw, pitch = math.radians(self.yaw_deg), math.radians(self.pitch_deg)
        forward = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
        right = np.cross(forward, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return np.stack([right, down, forward])

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, float) - self.position) @ self.rotation().T

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pc = self.to_camera(points)
        z = np.maximum(pc[:, 2], 1e-6)
        uv = np.stack([self.f * pc[:, 0] / z + self.width / 2.0, self.f * pc[:, 1] / z + self.height / 2.0], axis=1)
        return uv, pc[:, 2]


def look_at(position, target, f, width, height) -> Camera:
    d = np.asarray(target, float) - np.asarray(position, float)
    yaw = math.degrees(math.atan2(d[0], d[1]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return Camera(f, width, height, np.asarray(position, float), yaw, pitch)


# --------------------------------------------------------------------------------------------
# House geometry
# --------------------------------------------------------------------------------------------

@dataclass
class Surface:
    kind: str                 # "roof" | "wall" | "opening"
    verts: np.ndarray         # (N, 3) planar polygon
    color: tuple[int, int, int]
    name: str = ""
    plane_id: int = -1        # roof planes only

    def normal(self) -> np.ndarray:
        v = self.verts
        n = np.zeros(3)
        for i in range(len(v)):                       # Newell's method: robust for any planar polygon
            a, b = v[i], v[(i + 1) % len(v)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
        return n / (np.linalg.norm(n) or 1.0)


def _roof(verts, name, pid) -> Surface:
    return Surface("roof", np.array(verts, float), (68, 78, 150), name, pid)


def build_house(roof: str, W=12.0, D=8.0, Hw=3.0, pitch_deg=30.0, overhang=0.4,
                wing_w=5.0, wing_d=3.0, wing_x=2.0) -> list[Surface]:
    """Surfaces of one house. X is along the street, Y is depth (camera sits at negative Y), Z is up."""
    t = math.tan(math.radians(pitch_deg))
    o = overhang
    x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2
    wall = (190, 215, 225)
    surfaces: list[Surface] = []

    # walls (outward normals do not matter: the z-buffer sorts everything)
    surfaces += [
        Surface("wall", np.array([[x0, y0, 0], [x1, y0, 0], [x1, y0, Hw], [x0, y0, Hw]], float), wall, "front"),
        Surface("wall", np.array([[x1, y0, 0], [x1, y1, 0], [x1, y1, Hw], [x1, y0, Hw]], float), (170, 195, 205), "right"),
        Surface("wall", np.array([[x0, y1, 0], [x0, y0, 0], [x0, y0, Hw], [x0, y1, Hw]], float), (170, 195, 205), "left"),
        Surface("wall", np.array([[x1, y1, 0], [x0, y1, 0], [x0, y1, Hw], [x1, y1, Hw]], float), wall, "back"),
    ]
    # openings, a hair in front of their wall: these give the vanishing points something to hold on to
    e = 0.02
    for cx in (-4.0, -1.5, 3.5):
        surfaces.append(Surface("opening", np.array([[cx - 0.6, y0 - e, 1.0], [cx + 0.6, y0 - e, 1.0], [cx + 0.6, y0 - e, 2.3], [cx - 0.6, y0 - e, 2.3]], float), (120, 95, 70)))
    surfaces.append(Surface("opening", np.array([[0.6, y0 - e, 0], [1.6, y0 - e, 0], [1.6, y0 - e, 2.1], [0.6, y0 - e, 2.1]], float), (60, 70, 110)))
    for cy in (-2.0, 1.5):
        surfaces.append(Surface("opening", np.array([[x1 + e, cy - 0.6, 1.0], [x1 + e, cy + 0.6, 1.0], [x1 + e, cy + 0.6, 2.3], [x1 + e, cy - 0.6, 2.3]], float), (120, 95, 70)))
        surfaces.append(Surface("opening", np.array([[x0 - e, cy - 0.6, 1.0], [x0 - e, cy + 0.6, 1.0], [x0 - e, cy + 0.6, 2.3], [x0 - e, cy - 0.6, 2.3]], float), (120, 95, 70)))

    ze = Hw - o * t                       # eave height once the overhang drops below the wall plate
    if roof == "gable":
        zr = Hw + (D / 2) * t
        surfaces += [
            _roof([[x0 - o, y0 - o, ze], [x1 + o, y0 - o, ze], [x1 + o, 0, zr], [x0 - o, 0, zr]], "front", 0),
            _roof([[x1 + o, y1 + o, ze], [x0 - o, y1 + o, ze], [x0 - o, 0, zr], [x1 + o, 0, zr]], "back", 1),
        ]
        for xs in (x0, x1):               # gable-end wall triangles
            surfaces.append(Surface("wall", np.array([[xs, y0, Hw], [xs, y1, Hw], [xs, 0, zr]], float), (170, 195, 205), "gable end"))
    elif roof == "hip":
        zr = Hw + (D / 2) * t
        rx0, rx1 = x0 + D / 2, x1 - D / 2
        surfaces += [
            _roof([[x0 - o, y0 - o, ze], [x1 + o, y0 - o, ze], [rx1, 0, zr], [rx0, 0, zr]], "front", 0),
            _roof([[x1 + o, y1 + o, ze], [x0 - o, y1 + o, ze], [rx0, 0, zr], [rx1, 0, zr]], "back", 1),
            _roof([[x0 - o, y1 + o, ze], [x0 - o, y0 - o, ze], [rx0, 0, zr]], "left", 2),
            _roof([[x1 + o, y0 - o, ze], [x1 + o, y1 + o, ze], [rx1, 0, zr]], "right", 3),
        ]
    elif roof == "cross":
        # main gable along X (no overhang, so the valleys stay exact) plus a front-facing wing
        zr = Hw + (D / 2) * t
        wx0, wx1, wxc = wing_x - wing_w / 2, wing_x + wing_w / 2, wing_x
        wy = y0 - wing_d
        zw = Hw + (wing_w / 2) * t
        yv = y0 + (wing_w / 2)            # where the wing ridge dies into the main front plane
        surfaces += [
            _roof([[x0, y0, Hw], [wx0, y0, Hw], [wxc, yv, zw], [wx1, y0, Hw], [x1, y0, Hw], [x1, 0, zr], [x0, 0, zr]], "main front", 0),
            _roof([[x1, y1, Hw], [x0, y1, Hw], [x0, 0, zr], [x1, 0, zr]], "main back", 1),
            _roof([[wx0, wy, Hw], [wxc, wy, zw], [wxc, yv, zw], [wx0, y0, Hw]], "wing left", 2),
            _roof([[wxc, wy, zw], [wx1, wy, Hw], [wx1, y0, Hw], [wxc, yv, zw]], "wing right", 3),
        ]
        for xs in (x0, x1):
            surfaces.append(Surface("wall", np.array([[xs, y0, Hw], [xs, y1, Hw], [xs, 0, zr]], float), (170, 195, 205), "gable end"))
        surfaces += [
            Surface("wall", np.array([[wx0, wy, 0], [wx1, wy, 0], [wx1, wy, Hw], [wx0, wy, Hw]], float), wall, "wing front"),
            Surface("wall", np.array([[wx0, wy, Hw], [wx1, wy, Hw], [wxc, wy, zw]], float), wall, "wing gable"),
            Surface("wall", np.array([[wx0, y0, 0], [wx0, wy, 0], [wx0, wy, Hw], [wx0, y0, Hw]], float), (170, 195, 205), "wing left wall"),
            Surface("wall", np.array([[wx1, wy, 0], [wx1, y0, 0], [wx1, y0, Hw], [wx1, wy, Hw]], float), (170, 195, 205), "wing right wall"),
            Surface("opening", np.array([[wxc - 0.7, wy - e, 0.9], [wxc + 0.7, wy - e, 0.9], [wxc + 0.7, wy - e, 2.3], [wxc - 0.7, wy - e, 2.3]], float), (120, 95, 70)),
        ]
    else:
        raise ValueError(roof)
    return surfaces


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

@dataclass
class Scene:
    name: str
    image: np.ndarray                     # BGR
    camera: Camera
    planes: list[dict]                    # {"id", "name", "polygon": [[x, y], ...], "true_pitch_deg"}
    plane_regions: dict[int, np.ndarray]  # plane id -> visible region (bool), trees ignored
    tree_mask: np.ndarray                 # bool: everything standing in front of the house
    shared_edges: list[tuple[int, int]] = field(default_factory=list)
    solid_mask: Optional[np.ndarray] = None   # of those, the solid ones (car bodies, palm trunks)
    leafy_mask: Optional[np.ndarray] = None   # and the leafy ones (canopies, fronds)


def _zbuffer(surfaces: list[Surface], cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    H, W = cam.height, cam.width
    zbuf = np.full((H, W), np.inf, np.float32)
    ids = np.full((H, W), -1, np.int32)
    R = cam.rotation()
    for si, s in enumerate(surfaces):
        uv, z = cam.project(s.verts)
        if np.any(z < 0.3):
            continue
        n_c = R @ s.normal()
        d_c = float(n_c @ cam.to_camera(s.verts[:1])[0])
        x0, y0 = np.floor(uv.min(0)).astype(int) - 1
        x1, y1 = np.ceil(uv.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [np.round(uv - [x0, y0]).astype(np.int32)], 1)
        ys, xs = np.nonzero(mask)
        rays = np.stack([(xs + x0 - W / 2.0) / cam.f, (ys + y0 - H / 2.0) / cam.f, np.ones(len(xs))], axis=1)
        denom = rays @ n_c
        ok = np.abs(denom) > 1e-9
        depth = np.where(ok, d_c / np.where(ok, denom, 1.0), np.inf).astype(np.float32)
        yy, xx = ys + y0, xs + x0
        closer = (depth > 0) & (depth < zbuf[yy, xx])
        zbuf[yy[closer], xx[closer]] = depth[closer]
        ids[yy[closer], xx[closer]] = si
    return ids, zbuf


def _drop_collinear(polygon: np.ndarray, tol_px: float) -> np.ndarray:
    """Remove vertices that sit within `tol_px` of the chord between their neighbours."""
    pts = [p for p in polygon]
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            chord = c - a
            n = np.linalg.norm(chord)
            d = abs(chord[0] * (b[1] - a[1]) - chord[1] * (b[0] - a[0])) / n if n > 1e-9 else np.linalg.norm(b - a)
            if d < tol_px:
                del pts[i]
                changed = True
                break
    return np.array(pts)


def _draw_tree(image: np.ndarray, center: tuple[int, int], radius: int, rng) -> np.ndarray:
    H, W = image.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    cx, cy = center
    cv2.rectangle(image, (cx - radius // 10, cy), (cx + radius // 10, H), (40, 62, 85), cv2.FILLED)
    for _ in range(40):
        a, r = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0, 1)) * 0.75
        p = (int(cx + r * math.cos(a)), int(cy + r * math.sin(a) * 0.85))
        rr = int(radius * rng.uniform(0.18, 0.34))
        cv2.circle(image, p, rr, (int(rng.uniform(25, 60)), int(rng.uniform(95, 150)), int(rng.uniform(35, 75))), cv2.FILLED)
        cv2.circle(mask, p, rr, 1, cv2.FILLED)
    return mask.astype(bool)


def _draw_palm(image: np.ndarray, base_x: int, top: tuple[int, int], trunk_w: int, crown_r: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """A palm: a gently curved, tapering trunk from the ground to `top`, and a crown of drooping fronds."""
    H, W = image.shape[:2]
    trunk, crown = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    tx, ty = top
    bend = rng.uniform(-0.08, 0.08) * (H - ty)
    left, right = [], []
    for t in np.linspace(0, 1, 40):
        x = base_x + (tx - base_x) * t + bend * math.sin(math.pi * t)
        y = H + (ty - H) * t
        w = trunk_w * (1.0 - 0.35 * t)
        left.append((x - w / 2, y)); right.append((x + w / 2, y))
    poly = np.array(left + right[::-1], np.int32)
    cv2.fillPoly(image, [poly], (92, 118, 138)); cv2.fillPoly(trunk, [poly], 1)
    for k in range(int(rng.integers(9, 13))):
        a = rng.uniform(0, 2 * math.pi)
        length = crown_r * rng.uniform(0.75, 1.1)
        pts = []
        for t in np.linspace(0, 1, 12):
            droop = 0.55 * length * t * t
            pts.append((tx + math.cos(a) * length * t, ty + math.sin(a) * length * t * 0.6 + droop))
        pts = np.array(pts, np.int32)
        thick = max(3, int(crown_r * rng.uniform(0.09, 0.14)))
        colour = (int(rng.uniform(30, 60)), int(rng.uniform(110, 160)), int(rng.uniform(50, 80)))
        cv2.polylines(image, [pts], False, colour, thick, cv2.LINE_AA); cv2.polylines(crown, [pts], False, 1, thick)
    return trunk > 0, crown > 0


def _draw_car(image: np.ndarray, cx: int, top: int, width: int, rng) -> np.ndarray:
    """A car seen from the side: a body, a cabin, dark windows and wheels. Solid, and not green."""
    H, W = image.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    h = int(width * 0.42)
    colour = [(205, 205, 210), (40, 40, 170), (35, 35, 38), (160, 110, 60)][int(rng.integers(0, 4))]
    x0, x1 = cx - width // 2, cx + width // 2
    body = np.array([[x0, top + int(h * 0.42)], [x1, top + int(h * 0.45)], [x1, top + h], [x0, top + h]], np.int32)
    cabin = np.array([[x0 + int(width * 0.2), top + int(h * 0.44)], [x0 + int(width * 0.33), top], [x0 + int(width * 0.7), top],
                      [x0 + int(width * 0.84), top + int(h * 0.46)]], np.int32)
    for poly in (body, cabin):
        cv2.fillPoly(image, [poly], colour); cv2.fillPoly(mask, [poly], 1)
    window = np.array([[x0 + int(width * 0.25), top + int(h * 0.4)], [x0 + int(width * 0.35), top + int(h * 0.08)],
                       [x0 + int(width * 0.68), top + int(h * 0.08)], [x0 + int(width * 0.78), top + int(h * 0.42)]], np.int32)
    cv2.fillPoly(image, [window], (60, 50, 45))
    for wx in (x0 + int(width * 0.2), x0 + int(width * 0.8)):
        cv2.circle(image, (wx, top + h), int(h * 0.2), (25, 25, 25), cv2.FILLED); cv2.circle(mask, (wx, top + h), int(h * 0.2), 1, cv2.FILLED)
    return mask > 0


def render_scene(name: str, roof: str, cam_pos, target=(0.0, 0.0, 3.2), f=1500.0, size=(1600, 1000),
                 pitch_deg=30.0, tree: Optional[tuple[float, float, float]] = None, seed=0,
                 obstacles: Optional[list[dict]] = None, **house) -> Scene:
    """`obstacles`: extra things in front of the house, drawn after `tree`, each a dict with "type":
         tree  x, y, r              (fractions of the image width / height)
         palm  base_x, top_x, top_y, trunk_w, crown_r
         car   x, top, w"""
    rng = np.random.default_rng(seed)
    W, H = size
    cam = look_at(cam_pos, target, f, W, H)
    surfaces = build_house(roof, pitch_deg=pitch_deg, **house)
    ids, _ = _zbuffer(surfaces, cam)

    # background: sky gradient above the horizon, lawn below
    image = np.zeros((H, W, 3), np.uint8)
    horizon = int(np.clip(H / 2 + cam.f * math.tan(math.radians(cam.pitch_deg)), 0, H))
    ramp = np.linspace(0, 1, max(horizon, 1))[:, None, None]
    image[:horizon] = (np.array([235, 185, 125]) * (1 - ramp) + np.array([245, 228, 210]) * ramp).astype(np.uint8)
    ramp = np.linspace(0, 1, max(H - horizon, 1))[:, None, None]
    image[horizon:] = (np.array([70, 150, 80]) * (1 - ramp) + np.array([50, 112, 58]) * ramp).astype(np.uint8)

    light = np.array([0.35, -0.5, 0.8]); light /= np.linalg.norm(light)
    for si, s in enumerate(surfaces):
        region = ids == si
        if not region.any():
            continue
        shade = 0.55 + 0.45 * abs(float(s.normal() @ light))
        image[region] = np.clip(np.array(s.color) * shade, 0, 255).astype(np.uint8)

    # tile rows (parallel to the eave) and ridge / hip caps, drawn only where their plane is visible
    roofs = [(si, s) for si, s in enumerate(surfaces) if s.kind == "roof"]
    layer = image.copy()
    for si, s in roofs:
        n = s.normal()
        if n[2] < 0:
            n = -n
        eave_dir = np.cross([0, 0, 1.0], n); eave_dir /= np.linalg.norm(eave_dir)
        up_slope = np.cross(n, eave_dir)
        low = s.verts[np.argmin(s.verts[:, 2])]
        span = float(np.max((s.verts - low) @ up_slope))
        tmp = np.zeros((H, W), np.uint8)
        for k in np.arange(0.35, span, 0.35):
            a, b = low + k * up_slope - 30 * eave_dir, low + k * up_slope + 30 * eave_dir
            uv, z = cam.project(np.array([a, b]))
            if np.all(z > 0.5):
                cv2.line(tmp, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), 1, 1, cv2.LINE_AA)
        rows = (tmp > 0) & (ids == si)
        layer[rows] = (layer[rows] * 0.70).astype(np.uint8)
    for si, s in enumerate(surfaces):                     # siding: faint horizontal courses on every wall
        if s.kind != "wall" or not (ids == si).any():
            continue
        n = s.normal()
        along = np.cross([0, 0, 1.0], n)
        if np.linalg.norm(along) < 1e-6:
            continue
        along /= np.linalg.norm(along)
        base = s.verts[np.argmin(s.verts[:, 2])]
        tmp = np.zeros((H, W), np.uint8)
        for z in np.arange(0.3, float(s.verts[:, 2].max() - base[2]), 0.3):
            a, b = base + [0, 0, z] - 30 * along, base + [0, 0, z] + 30 * along
            uv, zc = cam.project(np.array([a, b]))
            if np.all(zc > 0.5):
                cv2.line(tmp, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), 1, 1, cv2.LINE_AA)
        courses = (tmp > 0) & (ids == si)
        layer[courses] = (layer[courses] * 0.74).astype(np.uint8)
    shared: list[tuple[int, int]] = []
    for ai in range(len(roofs)):
        for bi in range(ai + 1, len(roofs)):
            (sa_i, sa), (sb_i, sb) = roofs[ai], roofs[bi]
            for i in range(len(sa.verts)):
                a0, a1 = sa.verts[i], sa.verts[(i + 1) % len(sa.verts)]
                for j in range(len(sb.verts)):
                    b0, b1 = sb.verts[j], sb.verts[(j + 1) % len(sb.verts)]
                    if (np.allclose(a0, b1, atol=1e-6) and np.allclose(a1, b0, atol=1e-6)) or (np.allclose(a0, b0, atol=1e-6) and np.allclose(a1, b1, atol=1e-6)):
                        uv, z = cam.project(np.array([a0, a1]))
                        tmp = np.zeros((H, W), np.uint8)
                        cv2.line(tmp, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), 1, 5, cv2.LINE_AA)
                        cap = (tmp > 0) & ((ids == sa_i) | (ids == sb_i))
                        layer[cap] = np.clip(layer[cap] * 1.45 + 18, 0, 255).astype(np.uint8)
                        shared.append((sa.plane_id, sb.plane_id))
    image = layer

    # ground truth: visible roof planes as polygons
    planes, regions = [], {}
    for si, s in roofs:
        region = ids == si
        if region.sum() < 0.0005 * W * H:
            continue
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # slivers of far planes are dropped: nobody would annotate them
        cs = [c for c in cs if cv2.contourArea(c) >= 0.0005 * W * H and min(cv2.minAreaRect(c)[1]) >= 0.012 * W]
        if not cs:
            continue
        uv, _ = cam.project(s.verts)
        full = np.zeros((H, W), np.uint8)
        cv2.fillPoly(full, [np.round(uv).astype(np.int32)], 1)
        if len(cs) == 1 and region.sum() >= 0.985 * max(int(full.sum()), 1):
            polygons = [uv]
        else:                                    # partly hidden by another surface: annotate each visible piece
            polygons = [_drop_collinear(cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).astype(float), 2.0) for c in cs]
        n = s.normal()
        for k, polygon in enumerate(polygons):
            planes.append({"id": s.plane_id, "name": s.name + (f" (piece {k + 1})" if len(polygons) > 1 else ""),
                           "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
                           "true_pitch_deg": round(math.degrees(math.acos(min(1.0, abs(float(n[2]))))), 2)})
        kept_region = np.zeros((H, W), np.uint8)
        cv2.drawContours(kept_region, cs, -1, 1, thickness=cv2.FILLED)
        regions[s.plane_id] = region & (kept_region > 0)
    visible = {p["id"] for p in planes}
    shared = sorted({tuple(sorted(e)) for e in shared if e[0] in visible and e[1] in visible})

    tree_mask = np.zeros((H, W), bool)
    if tree is not None:
        tree_mask = _draw_tree(image, (int(tree[0] * W), int(tree[1] * H)), int(tree[2] * W), rng)
    solid, leafy = np.zeros((H, W), bool), tree_mask.copy()
    for ob in obstacles or []:
        if ob["type"] == "tree":
            leafy |= _draw_tree(image, (int(ob["x"] * W), int(ob["y"] * H)), int(ob["r"] * W), rng)
        elif ob["type"] == "palm":
            trunk, crown = _draw_palm(image, int(ob["base_x"] * W), (int(ob["top_x"] * W), int(ob["top_y"] * H)),
                                      int(ob["trunk_w"] * W), int(ob["crown_r"] * W), rng)
            solid |= trunk; leafy |= crown
        elif ob["type"] == "car":
            solid |= _draw_car(image, int(ob["x"] * W), int(ob["top"] * H), int(ob["w"] * W), rng)
    if obstacles:
        tree_mask = solid | leafy

    noise = rng.normal(0, 4.0, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return Scene(name, image, cam, planes, regions, tree_mask, shared,
                 solid_mask=solid if obstacles else None, leafy_mask=leafy if obstacles else None)


# --------------------------------------------------------------------------------------------
# SAM-like degradation
# --------------------------------------------------------------------------------------------

def _wobble(region: np.ndarray, rng, amplitude: float) -> np.ndarray:
    H, W = region.shape
    def field_():
        coarse = rng.normal(0, 1, (H // 24 + 2, W // 24 + 2)).astype(np.float32)
        fine = cv2.resize(cv2.GaussianBlur(coarse, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_CUBIC)
        return fine / (fine.std() + 1e-6) * amplitude
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    warped = cv2.remap(region.astype(np.uint8) * 255, xx + field_(), yy + field_(), cv2.INTER_LINEAR)
    return warped > 127


def sam_like_instances(scene: Scene, seed=0, wobble=2.2, parent: Optional[tuple[int, int]] = None,
                       duplicate: Optional[int] = None, fuse: Optional[tuple[int, int]] = None) -> list[dict]:
    """Instances shaped like SAM 3 output for `scene`.

    parent     (a, b): add a lower-scored instance covering planes a and b together.
    duplicate  plane id: add a second, slightly different instance of that plane.
    fuse       (a, b): planes a and b arrive as ONE instance (a crease hidden inside a mask).
    """
    rng = np.random.default_rng(seed + 1000)
    H, W = scene.tree_mask.shape
    groups: list[tuple[list[int], float]] = []
    fused = set(fuse) if fuse else set()
    for pid in scene.plane_regions:
        if pid in fused:
            continue
        groups.append(([pid], float(rng.uniform(0.55, 0.75))))
    if fuse:
        groups.append((list(fuse), float(rng.uniform(0.55, 0.7))))
    if parent:
        groups.append((list(parent), float(rng.uniform(0.5, 0.58))))
    if duplicate is not None:
        groups.append(([duplicate], float(rng.uniform(0.5, 0.6))))

    instances = []
    for pids, score in groups:
        region = np.zeros((H, W), bool)
        for pid in pids:
            region |= scene.plane_regions[pid]
        region &= ~scene.tree_mask                       # a tree in front takes a bite out of the mask
        region = _wobble(region, rng, wobble)
        grow = int(rng.integers(-2, 4))                  # neighbours end up with gaps or overlaps
        if grow:
            k = np.ones((2 * abs(grow) + 1,) * 2, np.uint8)
            region = (cv2.dilate if grow > 0 else cv2.erode)(region.astype(np.uint8), k) > 0
        mask = region.astype(np.uint8) * 255
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            continue
        for _ in range(int(rng.integers(0, 4))):         # holes from tile texture
            i = int(rng.integers(len(xs)))
            cv2.ellipse(mask, (int(xs[i]), int(ys[i])), (int(rng.integers(3, 9)), int(rng.integers(3, 7))), float(rng.uniform(0, 180)), 0, 360, 0, cv2.FILLED)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = max(cs, key=cv2.contourArea).reshape(-1, 2)
        for _ in range(int(rng.integers(0, 3))):         # bites out of the edge
            bx, by = boundary[int(rng.integers(len(boundary)))]
            cv2.ellipse(mask, (int(bx), int(by)), (int(rng.integers(5, 14)), int(rng.integers(4, 9))), float(rng.uniform(0, 180)), 0, 360, 0, cv2.FILLED)
        instances.append({"mask": mask, "score": round(score, 4), "source": "synthetic", "planes": pids})
    order = rng.permutation(len(instances))
    return [instances[i] for i in order]


# --------------------------------------------------------------------------------------------
# The fixed evaluation set
# --------------------------------------------------------------------------------------------

SCENE_SPECS = [
    dict(name="synth_gable_front",    roof="gable", cam_pos=(1.5, -24.0, 1.7),  f=1500, pitch_deg=30),
    dict(name="synth_gable_oblique",  roof="gable", cam_pos=(14.0, -20.0, 1.7), f=1400, pitch_deg=35, sam=dict(duplicate=0)),
    dict(name="synth_hip_oblique",    roof="hip",   cam_pos=(13.0, -20.0, 1.7), f=1450, pitch_deg=30),
    dict(name="synth_hip_front_tree", roof="hip",   cam_pos=(-2.0, -24.0, 1.7), f=1500, pitch_deg=28, tree=(0.27, 0.40, 0.085)),
    dict(name="synth_hip_high",       roof="hip",   cam_pos=(16.0, -18.0, 9.0), f=1350, pitch_deg=32, sam=dict(parent=(0, 3))),
    dict(name="synth_hip_left_tree",  roof="hip",   cam_pos=(-15.0, -19.0, 2.0), f=1400, pitch_deg=26, tree=(0.74, 0.37, 0.07)),
    dict(name="synth_cross_oblique",  roof="cross", cam_pos=(12.0, -22.0, 1.8), f=1450, pitch_deg=33),
    dict(name="synth_cross_high",     roof="cross", cam_pos=(10.0, -20.0, 8.0), f=1300, pitch_deg=38, sam=dict(parent=(0, 3))),
    dict(name="synth_hip_fused",      roof="hip",   cam_pos=(12.0, -21.0, 6.0), f=1400, pitch_deg=30, sam=dict(fuse=(0, 3))),
    dict(name="synth_gable_steep",    roof="gable", cam_pos=(-13.0, -21.0, 1.6), f=1600, pitch_deg=45, tree=(0.80, 0.42, 0.06)),
]


# Sample houses with obstacles in front of the roofline: palms, cars and trees (not all of them green).
OBSTACLE_SPECS = [
    dict(name="synth_gable_palm", roof="gable", cam_pos=(1.5, -24.0, 1.7), f=1500, pitch_deg=30,
         obstacles=[dict(type="palm", base_x=0.33, top_x=0.36, top_y=0.30, trunk_w=0.018, crown_r=0.09)]),
    dict(name="synth_hip_car", roof="hip", cam_pos=(10.0, -17.0, 1.5), f=1300, pitch_deg=30,
         obstacles=[dict(type="car", x=0.30, top=0.45, w=0.26)]),
    dict(name="synth_cross_palms", roof="cross", cam_pos=(12.0, -22.0, 1.8), f=1450, pitch_deg=33,
         obstacles=[dict(type="palm", base_x=0.35, top_x=0.40, top_y=0.28, trunk_w=0.016, crown_r=0.08),
                    dict(type="palm", base_x=0.66, top_x=0.62, top_y=0.30, trunk_w=0.015, crown_r=0.075)]),
    dict(name="synth_hip_tree_car", roof="hip", cam_pos=(1.0, -22.0, 1.7), f=1450, pitch_deg=30,
         obstacles=[dict(type="tree", x=0.80, y=0.42, r=0.07), dict(type="car", x=0.32, top=0.46, w=0.22)]),
    dict(name="synth_gable_two_palms", roof="gable", cam_pos=(-12.0, -20.0, 1.7), f=1400, pitch_deg=32,
         obstacles=[dict(type="palm", base_x=0.30, top_x=0.28, top_y=0.30, trunk_w=0.017, crown_r=0.08),
                    dict(type="palm", base_x=0.72, top_x=0.76, top_y=0.32, trunk_w=0.016, crown_r=0.08)]),
]


def make_scene(spec: dict, seed: int) -> tuple[Scene, list[dict]]:
    spec = dict(spec)
    sam_kwargs = spec.pop("sam", {})
    scene = render_scene(seed=seed, **spec)
    return scene, sam_like_instances(scene, seed=seed, **sam_kwargs)


def sam_like_occluders(scene: Scene, seed: int) -> Optional[dict]:
    """The obstacle pass SAM would give for `scene` (wobbled outlines), or None when nothing stands in front."""
    if scene.solid_mask is None:
        return None
    rng = np.random.default_rng(seed + 5000)
    return {"solid": _wobble(scene.solid_mask, rng, 1.5), "leafy": _wobble(scene.leafy_mask, rng, 2.0)}


SAMPLES = {"synth_cross_high": "Plain house (no obstacles)", "synth_gable_palm": "Palm in front of a gable",
           "synth_hip_car": "Car in front of the eave", "synth_hip_tree_car": "Tree on a corner and a car",
           "synth_gable_two_palms": "Two palms over the corners", "synth_cross_palms": "Two palms over a cross roof",
           "synth_hip_front_tree": "Tree hiding a hip corner"}


def load_sample(name: str) -> tuple[Scene, list[dict], Optional[dict]]:
    """A sample house by name, generated exactly as in the evaluation set (same seeds)."""
    for i, spec in enumerate(SCENE_SPECS):
        if spec["name"] == name:
            scene, instances = make_scene(spec, seed=i)
            return scene, instances, None
    for i, spec in enumerate(OBSTACLE_SPECS):
        if spec["name"] == name:
            scene, instances = make_scene(spec, seed=100 + i)
            return scene, instances, sam_like_occluders(scene, 100 + i)
    raise KeyError(name)
