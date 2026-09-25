"""The phased pipeline. `prepare` is everything that depends only on the image and the SAM
instances (run once); `solve` is the interactive geometry that reruns on every slider tick.
Every phase leaves its artefacts on the result so it can be inspected on its own.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import cleanup, directions, fitting, graph, interior, partition, roles, vanishing
from .config import Params


@dataclass
class Prepared:
    shape: tuple[int, int]
    partition: partition.Partition
    labels: np.ndarray                       # cleaned hard partition
    foliage: np.ndarray
    boundaries: list[cleanup.Boundary]
    ignore_mask: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    timings: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)   # later phases cache per-image work here (LSD, VPs)


def prepare(image: np.ndarray, instances: list[dict], params: Params, ignore_mask: Optional[np.ndarray] = None,
            foliage_override: Optional[np.ndarray] = None) -> Prepared:
    shape = image.shape[:2]
    t0 = time.perf_counter()
    part = partition.build_partition(instances, shape, params)                       # phase 1
    t1 = time.perf_counter()
    labels = cleanup.clean_partition(part, params)                                   # phase 2
    labels = partition.close_gaps(labels, int(params.gap_close_frac * max(shape)) | 1)
    foliage = cleanup.foliage_mask(image, params) if foliage_override is None else cleanup.dilate_mask(foliage_override, params)
    boundaries = cleanup.extract_boundaries(labels, params, foliage=foliage)
    t2 = time.perf_counter()
    exclude = foliage if ignore_mask is None else (foliage | (ignore_mask > 0))       # phase 4a, once per image
    segments = vanishing.detect_segments(image, params, exclude)
    vps = vanishing.estimate(segments, shape, params)
    t3 = time.perf_counter()
    return Prepared(shape, part, labels, foliage, boundaries, ignore_mask, image,
                    {"phase1_ms": round((t1 - t0) * 1000, 1), "phase2_ms": round((t2 - t1) * 1000, 1), "phase4a_vp_ms": round((t3 - t2) * 1000, 1)},
                    {"lsd": segments, "vps": vps})


def solve(prep: Prepared, params: Params, stop_after: int = 7, focal_px: Optional[float] = None, focal_source: Optional[str] = None) -> dict:
    long_edge = max(prep.shape)
    t0 = time.perf_counter()
    polygons = fitting.fit_planes(prep.boundaries, params, long_edge)                # phase 3
    timings = {"phase3_ms": round((time.perf_counter() - t0) * 1000, 2)}
    stage, info = 3, {}
    if stop_after >= 4:                                                              # phase 4b
        t1 = time.perf_counter()
        info["directions"] = directions.assign_priors(polygons, prep.extras.get("vps"), params)
        directions.apply_priors(polygons, params)
        polygons = [p for p in polygons if fitting.close_polygon(p, params, long_edge)]
        timings["phase4_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        stage = 4
    roof_graph = None
    if stop_after >= 5 and polygons:                                                 # phase 5
        t1 = time.perf_counter()
        roof_graph = graph.build(polygons, params, long_edge)
        before = graph.coverage(roof_graph, prep.labels)
        graph.optimise(roof_graph, prep.extras.get("vps"), params)
        coalesced = graph.coalesce(roof_graph, max(2.0, params.junction_coalesce_frac * long_edge), params)
        roof_graph.coverage = {"before": before, "after": graph.coverage(roof_graph, prep.labels)}
        trace = roof_graph.energy_trace
        info["graph"] = {"lines": len(roof_graph.lines), "junctions": len(roof_graph.junctions), "coalesced": coalesced, "rounds": len(trace) - 1,
                         "energy_start": round(trace[0]["total"], 1), "energy_end": round(trace[-1]["total"], 1),
                         "energy_trace": [{k: (round(v, 1) if k != "round" else v) for k, v in e.items()} for e in trace],
                         "coverage": roof_graph.coverage}
        timings["phase5_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        stage = 5
    if roof_graph is not None:
        segments, planes = graph.segments(roof_graph, params, long_edge), graph.plane_polygons(roof_graph)
    else:
        segments = fitting.polygons_to_segments(polygons, max(4.0, params.line_offset_frac * long_edge), params.merge_angle_deg)
        planes = [{"plane": p.plane, "part": p.part, "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in p.vertices()]} for p in polygons]
    vps = prep.extras.get("vps")
    if stop_after >= 6 and roof_graph is not None:                                   # phase 6
        t1 = time.perf_counter()
        planes, segments, info["creases"] = interior.find_creases(planes, segments, prep.extras.get("lsd", np.zeros((0, 4))), vps, prep.shape, params)
        timings["phase6_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        stage = 6

    t1 = time.perf_counter()                                                         # phase 7 (roles are cheap: every stage gets them)
    roles.assign(planes, segments, prep.labels, vps, params)
    for i, seg in enumerate(segments):
        seg["id"] = i
    if stop_after >= 7:
        if focal_px is None and vps is not None and vps.focal_px:
            focal_px, focal_source = vps.focal_px, "estimated from vanishing points"
        pitch = roles.calibrated_pitch(planes, segments, vps, focal_px, prep.shape, params)
        pitch["focal_source"] = focal_source if focal_px else None
        for plane in planes:
            sloped = [sg for sg in segments if plane["plane"] in sg.get("planes", []) and sg["role"] in ("rake", "hip", "valley") and sg.get("fitted", True)]
            if sloped:
                weights = np.array([sg["length_px"] for sg in sloped])
                angles = np.abs([sg["angle_deg"] for sg in sloped])
                order = np.argsort(angles)
                apparent = float(angles[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)])
                plane["pitch_apparent_deg"] = round(apparent, 2)
                plane["pitch_apparent"] = f"{math.tan(math.radians(min(apparent, 80.0))) * 12:.1f}:12"
                plane["pitch_apparent_standard"] = roles.standard_pitch(apparent, params.pitch_sigma_apparent_deg)
            calibrated = pitch["planes"].get(plane["plane"])
            if calibrated:
                plane["pitch_calibrated"] = calibrated
        info["pitch"] = {"focal_px": None if focal_px is None else round(float(focal_px), 1), "focal_source": pitch["focal_source"],
                         "calibrated_planes": len(pitch["planes"]), "roof": pitch.get("roof"),
                         "note": "apparent pitch is the slope as seen in the photo; calibrated pitch is the true roof angle and needs a focal length"}
        stage = 7
    timings["phase7_ms"] = round((time.perf_counter() - t1) * 1000, 2)
    timings["interactive_ms"] = round(sum(v for k, v in timings.items() if k[:6] in ("phase3", "phase4", "phase5", "phase6", "phase7")), 2)
    return {"segments": segments, "planes": planes, "polygons": polygons, "graph": roof_graph, "stage": stage, "info": info,
            "timings": {**prep.timings, **timings}}


def run(image: np.ndarray, instances: list[dict], params: Params, ignore_mask: Optional[np.ndarray] = None, stop_after: int = 7,
        focal_px: Optional[float] = None, focal_source: Optional[str] = None) -> dict:
    return solve(prepare(image, instances, params, ignore_mask), params, stop_after, focal_px, focal_source)
