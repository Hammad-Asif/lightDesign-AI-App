"""Every threshold in the pipeline, in one place, so the outer tuning loop can search them."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent.parent / "params.json"


@dataclass
class Params:
    # ---- Phase 1: hard partition -------------------------------------------------------
    score_floor: float = 0.35         # soft floor: instances under it lose priority, they are not dropped
    floor_penalty: float = 4.0        # effective-area multiplier applied under the floor
    score_gamma: float = 1.0          # priority = area / score**gamma  (smallest wins, so children beat parents)
    min_keep_fraction: float = 0.30   # an instance keeping this share of its mask is never a duplicate
    min_solid_fraction: float = 0.10  # below that, it survives only if this share remains after rims are opened away
    gap_close_frac: float = 0.006     # gaps between planes narrower than this (x long edge) are split between them

    # ---- Phase 2: cleanup and trusted arcs ---------------------------------------------
    close_frac: float = 0.008         # closing kernel, fraction of the long edge
    min_blob_frac: float = 0.05       # blobs under this share of a plane's largest blob are dropped
    min_instance_frac: float = 0.0002 # planes under this share of the image are noise
    notch_min_depth_frac: float = 0.01
    notch_max_mouth_ratio: float = 2.0
    notch_max_area_frac: float = 0.08
    foliage_exg: float = 0.06         # excess-green threshold for the colour-based foliage mask
    foliage_dilate_frac: float = 0.006
    adjacency_reach_px: int = 3       # a boundary pixel is an adjacency pixel if another plane lies this close
    straight_rms_px: float = 3.0      # a foliage-adjacent boundary piece this straight is a true edge, not an occluder outline
    straight_min_len_frac: float = 0.04  # ...provided it is at least this long (share of the plane perimeter, 30 px minimum)

    # ---- Phase 3: per-plane line fitting -----------------------------------------------
    epsilon: float = 0.010            # corner-seed tolerance, fraction of the largest plane perimeter (slider)
    eps_cap_own_frac: float = 0.04    # ...capped at this share of a small plane's own perimeter
    merge_angle_deg: float = 5.0      # adjacent edges closer than this in direction are one line (slider)
    min_trusted_frac: float = 0.35    # an edge needs this share of trusted pixels to be fitted at all
    min_trusted_px: int = 8
    huber_inlier_px: float = 2.5      # pixels farther than this from the Huber fit get zero weight afterwards
    corner_guard_frac: float = 0.02   # a visible corner may sit at most this far (x long edge) from its seed
    gap_guard_ratio: float = 1.5      # an occluded corner may sit up to this many gap lengths from the gap

    # ---- Phase 4: vanishing points and direction regularisation ------------------------
    lsd_max_dim: int = 1200
    lsd_min_len_frac: float = 0.015   # of the image diagonal
    vp_inlier_deg: float = 2.0
    vp_iterations: int = 1500
    vp_min_support_frac: float = 0.06 # of total segment length
    vp_snap_deg: float = 4.0          # a line this close to a VP direction is pulled onto it
    dir_cluster_deg: float = 3.0      # lines this close in 2D direction share one direction (fallback / hips)
    symmetry_deg: float = 4.0         # opposite rakes within this of mirror symmetry are made symmetric

    # ---- Phase 5: planar graph and the consistency loop --------------------------------
    snap_radius_frac: float = 0.008   # corners closer than this (x long edge) are one junction (slider)
    line_offset_frac: float = 0.006   # edges this close in offset (and merge_angle in direction) share a line
    w_fit: float = 1.0
    w_junc: float = 30.0              # one junction counts like this many boundary pixels
    w_dir: float = 3.0                # direction prior stiffness relative to the data (slider: regularity)
    w_par: float = 0.0                # shared-2-D-direction prior; hurts under perspective, so off unless VPs are unstable
    w_par_fallback: float = 2.0       # ...in which case it is the only direction prior and runs at this weight
    w_cov: float = 200.0              # weight of the coverage term (fraction of area) in the reported energy
    junction_ridge: float = 0.02      # anchored ridge in the junction solve (absolute, against unit normals); neutral here, needed on coarse masks
    junction_min_angle_deg: float = 0.0  # corners on lines closer than this to parallel never share a junction (0 = off)
    junction_coalesce_frac: float = 0.0025  # after the relaxation, junctions closer than this (x long edge) are one junction
    min_segment_frac: float = 0.0     # segments shorter than this share of the long edge are not emitted (0 = 1 px)
    max_rounds: int = 12
    energy_tol: float = 1e-3          # stop when the relative energy drop per round falls under this

    # ---- Phase 6: creases hidden inside one mask ---------------------------------------
    crease_min_len_frac: float = 0.04 # of the image diagonal
    crease_min_support: float = 0.6   # share of the hypothesised crease covered by agreeing LSD segments
    crease_perp_px: float = 3.0
    crease_angle_deg: float = 4.0
    crease_eave_parallel_deg: float = 10.0

    # ---- Phase 7: roles and pitch ------------------------------------------------------
    horizontal_vp_deg: float = 6.0    # an edge this close to a horizontal VP direction is "horizontal in plan"
    pitch_sigma_calibrated_deg: float = 2.0
    pitch_sigma_apparent_deg: float = 6.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Params":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: Path | None = None) -> "Params":
        path = path or PARAMS_FILE
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return cls()

    def save(self, path: Path | None = None) -> None:
        (path or PARAMS_FILE).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
