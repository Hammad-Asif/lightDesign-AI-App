"""Phase 4a: vanishing points from ALL lines in the photo.

Walls, window frames, door frames, garage doors and tile rows say far more about the scene's
horizontal and vertical directions than the roof outline does, so the VPs are estimated from
every LSD segment in the image (minus foliage and any ignore mask), never from roof lines alone.

Everything is done in homogeneous coordinates, so a VP at infinity (a facade seen head-on) is
handled like any other. Coordinates are centred and scaled before any cross product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import Params


@dataclass
class VanishingPoints:
    vertical: Optional[np.ndarray]            # homogeneous, in centred/scaled coordinates
    horizontals: list[np.ndarray]             # strongest first, at most two
    supports: dict = field(default_factory=dict)
    centre: np.ndarray = None                 # pixel
    scale: float = 1.0
    focal_px: Optional[float] = None          # from the orthogonality of the two horizontal VPs
    stable: bool = False

    def _norm(self, point_px: np.ndarray) -> np.ndarray:
        return (np.asarray(point_px, float) - self.centre) / self.scale

    def direction_at(self, vp: np.ndarray, point_px: np.ndarray) -> Optional[np.ndarray]:
        """Unit direction, at an image point, of the line through that point and the VP."""
        d = vp[:2] - vp[2] * self._norm(point_px)
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else None

    def named(self) -> list[tuple[str, np.ndarray]]:
        out = [("v", self.vertical)] if self.vertical is not None else []
        return out + [(f"h{i + 1}", h) for i, h in enumerate(self.horizontals)]

    def to_pixels(self, vp: np.ndarray) -> Optional[list[float]]:
        if abs(vp[2]) < 1e-9:
            return None
        return [float(vp[0] / vp[2] * self.scale + self.centre[0]), float(vp[1] / vp[2] * self.scale + self.centre[1])]

    def describe(self) -> dict:
        return {"stable": self.stable, "focal_px_from_vps": None if self.focal_px is None else round(self.focal_px, 1),
                "points": {name: {"pixel": self.to_pixels(vp), "support": round(self.supports.get(name, 0.0), 3),
                                  "direction_at_centre_deg": None if self.direction_at(vp, self.centre) is None else round(math.degrees(math.atan2(*self.direction_at(vp, self.centre)[::-1])) % 180, 2)}
                           for name, vp in self.named()}}


def detect_segments(image_bgr: np.ndarray, params: Params, exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """LSD segments (N, 4) in full-resolution pixels, long enough to carry a direction."""
    H, W = image_bgr.shape[:2]
    s = min(1.0, params.lsd_max_dim / max(H, W))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if s < 1:
        gray = cv2.resize(gray, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    segs = lines.reshape(-1, 4).astype(float) / s
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    segs = segs[length >= params.lsd_min_len_frac * math.hypot(H, W)]
    if exclude is not None and len(segs):
        keep = np.ones(len(segs), bool)
        for t in (0.25, 0.5, 0.75):
            x = np.clip(np.round(segs[:, 0] + t * (segs[:, 2] - segs[:, 0])).astype(int), 0, W - 1)
            y = np.clip(np.round(segs[:, 1] + t * (segs[:, 3] - segs[:, 1])).astype(int), 0, H - 1)
            keep &= ~exclude[y, x]
        segs = segs[keep]
    return segs


def _ransac_vp(lines, mids, dirs, weights, rng, iterations, inlier_deg):
    """Best VP for the given segments: (vp, inlier mask, support) or None."""
    n = len(lines)
    if n < 4:
        return None
    cos_tol = math.cos(math.radians(inlier_deg))
    prob = weights / weights.sum()
    best = (None, None, 0.0)
    for start in range(0, iterations, 256):
        k = min(256, iterations - start)
        i, j = rng.choice(n, k, p=prob), rng.choice(n, k, p=prob)
        vps = np.cross(lines[i], lines[j])
        norms = np.linalg.norm(vps, axis=1)
        ok = (norms > 1e-12) & (i != j)
        vps = vps[ok] / norms[ok, None]
        if not len(vps):
            continue
        d = vps[:, None, :2] - vps[:, None, 2:3] * mids[None, :, :]
        dn = np.linalg.norm(d, axis=2)
        cosang = np.abs((d * dirs[None, :, :]).sum(axis=2)) / np.maximum(dn, 1e-12)
        inliers = (cosang >= cos_tol) & (dn > 1e-9)
        scores = (inliers * weights[None, :]).sum(axis=1)
        b = int(np.argmax(scores))
        if scores[b] > best[2]:
            best = (vps[b], inliers[b], float(scores[b]))
    if best[0] is None:
        return None
    vp, inliers, _ = best
    for _ in range(2):                                    # least-squares refinement on the inliers
        if inliers.sum() < 2:
            break
        _, _, vt = np.linalg.svd(lines[inliers] * weights[inliers, None], full_matrices=False)
        vp = vt[-1]
        d = vp[None, :2] - vp[2] * mids
        dn = np.linalg.norm(d, axis=1)
        inliers = (np.abs((d * dirs).sum(axis=1)) / np.maximum(dn, 1e-12) >= cos_tol) & (dn > 1e-9)
    return vp, inliers, float(weights[inliers].sum())


def estimate(segments: np.ndarray, shape: tuple[int, int], params: Params, seed: int = 0) -> VanishingPoints:
    H, W = shape
    centre, scale = np.array([W / 2.0, H / 2.0]), max(H, W) / 2.0
    result = VanishingPoints(None, [], {}, centre, scale)
    if len(segments) < 8:
        return result
    rng = np.random.default_rng(seed)
    p1, p2 = (segments[:, :2] - centre) / scale, (segments[:, 2:] - centre) / scale
    lines = np.cross(np.column_stack([p1, np.ones(len(p1))]), np.column_stack([p2, np.ones(len(p2))]))
    lines /= np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
    mids = (p1 + p2) / 2.0
    dirs = p2 - p1
    weights = np.linalg.norm(dirs, axis=1)
    dirs = dirs / weights[:, None]
    total = float(weights.sum())

    # vertical first, from segments within 20 degrees of upright
    upright = np.abs(dirs[:, 1]) >= math.cos(math.radians(20))
    remaining = np.ones(len(segments), bool)
    if upright.sum() >= 4:
        found = _ransac_vp(lines[upright], mids[upright], dirs[upright], weights[upright], rng, params.vp_iterations // 2, params.vp_inlier_deg)
        if found and found[2] >= 0.5 * params.vp_min_support_frac * total:
            result.vertical = found[0]
            result.supports["v"] = found[2] / total
            idx = np.nonzero(upright)[0][found[1]]
            remaining[idx] = False

    # The two horizontal VPs are not searched for independently. The vertical VP fixes the horizon's
    # orientation; a focal length fixes its offset; and one horizontal VP on that horizon fixes the
    # other by orthogonality. So every hypothesis is a complete Manhattan frame, scored as a whole.
    pool = remaining
    if pool.sum() >= 6:
        L, M, D, Wt = lines[pool], mids[pool], dirs[pool], weights[pool]
        cos_tol = math.cos(math.radians(params.vp_inlier_deg))
        if result.vertical is not None and abs(result.vertical[2]) > 1e-9:
            vv = result.vertical[:2] / result.vertical[2]
            dist_v = float(np.linalg.norm(vv))
            n_h = vv / dist_v                                   # horizon normal, pointing at the vertical VP
        else:
            vd = result.vertical[:2] if result.vertical is not None else np.array([0.0, 1.0])
            n_h, dist_v = vd / np.linalg.norm(vd), math.inf
        e_h = np.array([-n_h[1], n_h[0]])

        def support(vps: np.ndarray) -> np.ndarray:             # (K, 3) -> (K, N) inlier matrix
            d = vps[:, None, :2] - vps[:, None, 2:3] * M[None, :, :]
            dn = np.linalg.norm(d, axis=2)
            return ((np.abs((d * D[None, :, :]).sum(axis=2)) / np.maximum(dn, 1e-12)) >= cos_tol) & (dn > 1e-9)

        prob = Wt / Wt.sum()
        k = min(400, params.vp_iterations)
        i, j = rng.choice(len(L), k, p=prob), rng.choice(len(L), k, p=prob)
        hyp = np.cross(L[i], L[j])
        hyp = hyp[(np.linalg.norm(hyp, axis=1) > 1e-12) & (i != j)]
        # where each hypothesis sits along the horizon (t may be infinite: keep it homogeneous)
        t_num, t_den = hyp[:, :2] @ e_h, hyp[:, 2]              # t = t_num / t_den
        best = (-1.0, None, None, None)
        for f in np.geomspace(0.5 * W, 3.0 * W, 14) / scale:
            offset = -f * f / dist_v if math.isfinite(dist_v) else 0.0
            h1 = np.column_stack([offset * n_h[0] * t_den + e_h[0] * t_num, offset * n_h[1] * t_den + e_h[1] * t_num, t_den])
            # orthogonality: offset^2 + t1 * t2 = -f^2  ->  t2 = -(f^2 + offset^2) / t1 = -(f^2 + offset^2) * t_den / t_num
            c2 = -(f * f + offset * offset)
            h2 = np.column_stack([offset * n_h[0] * t_num + e_h[0] * c2 * t_den, offset * n_h[1] * t_num + e_h[1] * c2 * t_den, t_num])
            h1 /= np.maximum(np.linalg.norm(h1, axis=1, keepdims=True), 1e-12)
            h2 /= np.maximum(np.linalg.norm(h2, axis=1, keepdims=True), 1e-12)
            in1, in2 = support(h1), support(h2)
            s1, s2 = (in1 * Wt).sum(axis=1), ((in2 & ~in1) * Wt).sum(axis=1)
            score = s1 + s2
            b = int(np.argmax(score))
            if score[b] > best[0]:
                best = (float(score[b]), (h1[b], float(s1[b])), (h2[b], float(s2[b])), f * scale)
        def refine(vp: np.ndarray) -> tuple[np.ndarray, float]:
            """The frame search finds the right line families; their VPs are then refit freely."""
            inl = support(vp[None, :])[0]
            for _ in range(3):
                if inl.sum() < 3:
                    break
                _, _, vt = np.linalg.svd(L[inl] * Wt[inl, None], full_matrices=False)
                vp = vt[-1]
                inl = support(vp[None, :])[0]
            return vp, float(Wt[inl].sum())

        if best[1] is not None and best[0] >= params.vp_min_support_frac * total:
            best = (best[0], refine(best[1][0]), refine(best[2][0]), best[3])
            ordered = sorted([best[1], best[2]], key=lambda v: -v[1])
            result.horizontals = [v for v, sup in ordered if sup > 0] or [ordered[0][0]]
            for idx, (v, sup) in enumerate(ordered[:len(result.horizontals)]):
                result.supports[f"h{idx + 1}"] = sup / total
            # Focal length: only meaningful when BOTH horizontal VPs are really observed. Prefer the
            # orthogonality of the freely refined VPs over the coarse grid value that found them.
            sup = [v[1] / total for v in ordered]
            # a VP close to the principal point (a facade seen head-on) makes the orthogonality product meaningless
            off_centre = all(abs(v[0][2]) < 1e-9 or np.linalg.norm(v[0][:2] / v[0][2]) > 0.3 for v in ordered)
            if len(sup) == 2 and min(sup) >= 0.08 and off_centre:
                h1, h2 = ordered[0][0], ordered[1][0]
                f_grid = best[3]
                if abs(h1[2]) > 1e-9 and abs(h2[2]) > 1e-9:
                    f2 = -float((h1[:2] / h1[2]) @ (h2[:2] / h2[2]))
                    f_free = math.sqrt(f2) * scale if f2 > 0 else None
                    result.focal_px = f_free if f_free and 0.4 * W <= f_free <= 4.0 * W else f_grid
                else:
                    result.focal_px = f_grid
    result.stable = result.vertical is not None and len(result.horizontals) >= 1
    return result
