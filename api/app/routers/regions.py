import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.models import Region as RegionRow
from app.providers.base import get_provider
from app.schemas import (
    CreateRegionRequest, DetectRequest, FindLinesRequest, PhotoDetail, RefineRequest, Region, RegionPatch, TargetTap,
)
from app.routers import photos as photos_router
from app.services import (
    camera, detect, drawn, fusion, intent, lighting, occlusion, partition, partition_reference, roofline, rooflines,
    snap, target,
)
from app.storage import Store, get_store, photo_dict, region_dict

log = logging.getLogger(__name__)

router = APIRouter(prefix="/photos/{photo_id}", tags=["regions"])

SIDE_LABEL = {"front": "front", "left": "left side", "right": "right side"}
EDGE_LABEL = {
    "eave": "Eave", "rake": "Rake", "hip": "Hip", "ridge": "Ridge", "valley": "Valley",
    "junction": "Roof meets wall", "unknown": "Unlabelled edge",
}


def _blocks(kind: str) -> bool:
    return kind in occlusion.BLOCKER_KINDS


def _key(span) -> tuple:
    return tuple(map(tuple, span[0]))


def roof_lines(
    roof_masks: list[np.ndarray],
    blockers: np.ndarray | None,
    parsed: intent.Intent,
    hybrid: bool,
    edges=None,
    # Blockers whose own outline is straight (occlusion.SOLID_KINDS): these are kept out of the
    # boundary fit. `blockers` — everything in front of the house — still hides and still joins.
    solid: np.ndarray | None = None,
    interior: bool = False,
    partitioned: bool = False,
    image=None,
    scores: list[float] | None = None,
    details: dict | None = None,
    junctions: bool = True,
    constrained: bool = True,
    unrequested: list | None = None,
    reference: dict | None = None,
    vp_region: np.ndarray | None = None,
) -> list[tuple]:
    """The roof lines detection returns for an instruction, as (polyline, side, edge).

    `edges` is the photo's gradient field from services/snap.field; with it, every line that
    survives is pulled onto the image edge it follows. `partitioned` swaps the per-plane
    extraction for services/partition (EDGE_EXTRACTOR=partition); everything after it is shared.
    That mode also reads the photo itself (`image`, for foliage and vanishing points) and the
    instances' scores, and fills `details` with the inferred connectors' keys and the pitch per
    plane; hybrid ignores all of them.
    """
    width = roof_masks[0].shape[1]
    short: set = set()
    seams: list[tuple] = []
    options: dict = {}
    if partitioned and reference is not None:
        # The reference implementation itself, on its own working copy: what the reference app
        # draws is what comes out here. Junction relabelling and constrained snap do not apply.
        planes, inferred, pitches, solved = partition_reference.result(
            reference["working_bgr"], reference["working_masks"], scores or [None] * len(roof_masks), roof_masks, width,
            reference["scale"], foliage=reference.get("foliage"), focal_35mm=reference.get("focal_35mm"),
        )
        if details is not None:
            details["inferred"], details["pitch"] = inferred, pitches
            details["reference"] = {"planes": len(solved["planes"]), "segments": len(solved["segments"]), "timings": solved["timings"]}
            details["planes"] = solved["plane_polygons"]
    elif partitioned:
        planes, inferred, pitches = partition.partition_result(
            # Only the straight-edged objects are kept out of the fit; everything in front of
            # the house still hides a line and still joins one back up, below.
            roof_masks, width, solid if solid is not None else blockers,
            image=image, scores=scores, junctions=junctions, constrained=constrained, vp_region=vp_region
        )
        if details is not None:
            details["inferred"], details["pitch"] = inferred, pitches
        # Snapping reach and smoothing follow the roof's size in the photo, not its pixel count.
        options = partition.snap_options(partition.roof_extent(roof_masks))
    else:
        planes = [roofline.plane_spans(mask, width, hybrid) for mask in roof_masks]
        # Pieces a palm or a car cut short are offered as well, so the two sides of a blocker can
        # be joined back up. They have to earn their place: anything still a stub after bridging goes.
        extra = [roofline.blocker_pieces(mask, width, blockers) for mask in roof_masks]
        short = {_key(span) for pieces in extra for span in pieces}
        planes = [plane + pieces for plane, pieces in zip(planes, extra)]
        # A lower roof's top edge that runs into the wall of the storey above is not a rake.
        planes = roofline.mark_junctions(planes, roof_masks, width)
        # Ridges where two planes meet. Only abutting masks produce one; a plane whose own
        # outline holds its ridge gets none, never a boundary segment called a ridge.
        seams = roofline.seam_ridges(roof_masks, width)
    # Hips inside a merged mask, if they can be confirmed against the photo.
    inside: list[tuple] = []
    if interior and edges is not None:
        gradients, floor = edges[:2], edges[2]
        inside = [h for mask in roof_masks for h in roofline.interior_hips(mask, width, gradients, floor)]

    if partitioned:
        # The partition graph already emits every edge exactly once, closed at shared junctions,
        # with occluded stretches bridged by its own inferred connectors. The hybrid path's
        # dedupe / join / bridge steps were tuned for per-plane outlines: on this output they
        # drop real lines (measured: 45 -> 34 spans, 838 px lost, the main roof's hip among
        # them) and give a joined polyline one role for several edges. So they are skipped.
        spans = [span for spans in planes for span in spans] + inside
        # Two things neither engine gets right on its own, corrected on the finished spans so
        # the reference engine is covered too: the walls of a notch the mask took around a palm
        # trunk drawn as lit "hips" (dropped, so the eave runs through the trunk), and one
        # edge drawn twice after the relaxation pulled two lines together.
        extent = partition.roof_extent(roof_masks)
        spans = partition.clean_spans(spans, extent)
        # A roof edge is one line; an object in front of it is something the line passes
        # behind. Pieces a detected trunk, tree or car cut apart are joined back into one
        # straight line, and an end such an object cut off is carried through to its far side
        # (the hybrid path's bridge, kept; its dedupe and join stay off).
        roof_union = np.logical_or.reduce(roof_masks) if len(roof_masks) > 1 else roof_masks[0]
        spans = partition.bridge_through(spans, blockers, width, extent, roof=roof_union)
    else:
        spans = roofline.dedupe([span for spans in planes for span in spans] + inside + seams, width)
        # Extraction splits an edge wherever it turns; a plane's pieces that meet end to end are rejoined.
        spans = roofline.join_pieces(spans, planes)

        # A line a detected object splits in two is joined back up across that object.
        spans = occlusion.bridge(spans, blockers, width)
        # A piece kept only because it stopped at a blocker, and still not joined to anything, is
        # the noise the minimum length exists to remove.
        spans = [s for s in spans if _key(s) not in short]

    def wanted(span) -> bool:
        # "unknown" means the edge could not be identified, not that it was excluded. Keep
        # it so the editor can flag it for review rather than silently drop a possible eave.
        # A junction is identified — roof meeting wall — so an outline request leaves it dark.
        if parsed.sides and span[1] not in parsed.sides:
            return False
        return not parsed.edges or span[2] in parsed.edges or span[2] == "unknown"

    if unrequested is not None and partitioned:
        # The partition graph found these too (a ridge on a "roofline" request, the other sides
        # on a "front" request). They are handed back so the editor can keep them as hidden
        # regions: detected like the reference method, lit only when asked for.
        unrequested.extend(s for s in spans if not wanted(s))
    spans = [s for s in spans if wanted(s)]

    # Last: the mask decides which lines exist and what they are; the photo decides exactly
    # where each one sits.
    if edges is not None:
        gx, gy, minimum = edges
        if partitioned:
            # Partition's segments are already straight fits. Where snapping finds a straight
            # edge to move to, take it; where it would only hand back a wobble traced through
            # brick courses or branches, the fitted segment stays.
            snapped = []
            for points, side, edge in spans:
                moved, straight = snap.fitted_line(points, (gx, gy), minimum, **options)
                snapped.append((moved if straight else points, side, edge))
            spans = snapped
        else:
            spans = snap.lines(spans, (gx, gy), minimum, **options)
    return spans


def compute_roof_lines(
    roof_masks: list[np.ndarray],
    scores: list[float],
    blockers: np.ndarray | None,
    solid: np.ndarray | None,
    parsed: intent.Intent,
    image,
    reference: dict | None = None,
    vp_region: np.ndarray | None = None,
) -> tuple[list[tuple], list[tuple], dict]:
    """roof_lines as /detect runs it under the current settings: (spans, unrequested, details).

    Shared by detection and by the target-house tap, which redraws the lines from stored plane
    masks, so both produce the same lines from the same planes.
    """
    settings = get_settings()
    partitioned = settings.edge_extractor == "partition"
    # The reference geometry is final after its relaxation; snapping on top is opt-in there.
    snapping = settings.snap_to_edges and (not partitioned or settings.partition_snap)
    details: dict = {}
    unrequested: list[tuple] = []
    spans = roof_lines(
        roof_masks,
        blockers,
        parsed,
        settings.edge_extractor == "hybrid",
        snap.field(image) if snapping or settings.interior_hips else None,
        solid=solid,
        interior=settings.interior_hips,
        partitioned=partitioned,
        image=image,
        scores=scores,
        details=details,
        junctions=settings.partition_junctions,
        constrained=settings.partition_constrained_snap,
        unrequested=unrequested,
        reference=reference,
        vp_region=vp_region,
    )
    return spans, unrequested, details


def _line_label(edge: str, side: str | None, number: int, suffix: str = "") -> str:
    if side is None:
        return f"{EDGE_LABEL.get(edge, edge)} {number}{suffix}"
    return f"{EDGE_LABEL.get(edge, edge)} — {SIDE_LABEL.get(side, side)} {number}{suffix}"


def roof_rows(result: fusion.Result, parsed: intent.Intent, found_in_box: bool = False) -> list[RegionRow]:
    """The region rows for the unified roof lines: lit lines first, then the ones kept for review.

    Labels, colours and order are the partition-only pipeline's own. What is added: `confidence`
    (also the `score` the editor shows, instead of a fixed 0.5) and `sources`, the one source the
    line came from. A line with no side
    on a side-specific instruction (V3 without a vanishing point) is kept, not lit, and says so.
    """
    rows: list[RegionRow] = []
    lit = unlit = 0
    for line, requested in zip(result.lines, result.requested):
        points, side, edge = line.span()
        if requested:
            lit += 1
            number, suffix = lit, " (found in box)" if found_in_box else ""
        else:
            unlit += 1
            number = unlit
            suffix = " (assign side and enable)" if parsed.sides and side is None else " (not lit)"
        rows.append(
            RegionRow(
                kind="roof",
                label=_line_label(edge, side, number, suffix),
                geometry="line",
                # A connector drawn across an occluded stretch from no pixels is kept as
                # data, marked so the editor can show it for what it is.
                source="inferred" if line.inferred else "auto",
                color=detect.edge_color(edge),
                score=line.confidence if line.confidence is not None else 0.5,
                confidence=line.confidence,
                sources=list(line.source),
                visible=requested,
                side=side,
                edge=edge,
                polyline=points,
                mask_url=None,
            )
        )
    return rows


def _roofline_extra(result: fusion.Result, image, store: Store, photo) -> dict:
    """What a unified run leaves on the photo: its report, the camera its vanishing points give
    (services/camera.py, what scale and placement are computed from), and with ROOFLINE_DEBUG
    its stages (plus overlay images in the photo's debug/ folder, served like any other file)."""
    debug = result.debug
    if debug is not None:
        debug["overlays"] = [f"/files/photos/{photo.id}/debug/{name}"
                             for name in fusion.write_overlays(image, debug, store.photo_dir(photo.id) / "debug")]
    return {"roofline": result.report, "roofline_debug": debug,
            "camera": camera.build(result.scene, image.size, photo.focal_35mm).to_dict()}


@router.post("/detect", response_model=list[Region])
def run_detection(photo_id: str, body: DetectRequest, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    image = store.load_image(photo)

    parsed = intent.Intent()
    if body.kinds:
        selected = body.kinds
    elif body.all_kinds:
        selected = None
    else:
        parsed = intent.parse(photo.prompt)
        selected = parsed.kinds or None

    # This commit also closes the read transaction before the slow detection call. An explicit
    # kinds list or all_kinds still records what the instruction itself asked for, so the
    # specification the user confirms is the instruction's, not the override's.
    store.update_photo(photo, interpreted=lighting.interpretation(photo.prompt, store.products()))

    settings = get_settings()
    # One detection path. SAM always runs: roof planes, blockers, the target house and every other
    # requested kind come from it. The roof lines come from V3 on the target-house crop when V3 is
    # the source (ROOFLINE_SOURCES) and answers; partition's lines only when it is the source or
    # V3 failed. services/fusion.py.
    roof_wanted = "roof" in (selected or detect.DEFAULT_KINDS)
    if settings.scale_doors and roof_wanted:
        # The scale's outside check: doors and garage doors of standard size (SCALE_DOORS).
        selected = list(dict.fromkeys([*(selected or detect.DEFAULT_KINDS), "door", "garage_door"]))
    use_v3 = "v3" in settings.roofline_sources and roof_wanted
    partitioned = settings.edge_extractor == "partition"
    clock = time.perf_counter()

    def lap(stage: str) -> None:
        """Where detection time goes, on the server log: the only place slowness is visible."""
        nonlocal clock
        now = time.perf_counter()
        log.info("detect %s: %s %.1fs", photo_id[:8], stage, now - clock)
        clock = now
    # Partition mode detects the way the reference method does: a 2000px working copy, 12
    # instances per call, JPEG. With the reference engine the working copy is built by the
    # reference's own code (OpenCV INTER_AREA), so SAM 3 sees exactly what the reference sends.
    use_reference = partitioned and settings.partition_engine == "reference"
    if use_reference and not partition_reference.available():
        # Said here, before a billed detection call, rather than from inside the geometry after it.
        raise HTTPException(503, partition_reference.MISSING)
    reference: dict | None = None
    if use_reference:
        working_bgr, small, scale = partition_reference.working_copy(image)
        reference = {"working_bgr": working_bgr, "scale": scale, "focal_35mm": None}
    else:
        small, _ = detect.working_copy(
            image, settings.partition_detect_max_edge if partitioned else settings.detect_max_edge
        )
    try:
        provider = get_provider()
        if partitioned and settings.segmentation_provider == "fal":
            from app.providers.fal import FalProvider

            provider = FalProvider(max_masks=settings.partition_fal_max_masks, image_format="jpeg")
        # The sky only bounds the top of V3's crop, so it is asked for only when V3 will run.
        detections = detect.run(small, provider, selected, settings.detect_blockers,
                                sky=use_v3 and settings.v3_crop_sky)
        if use_reference and settings.partition_tree_pass and hasattr(provider, "_call"):
            trees = np.zeros((small.height, small.width), bool)
            for _, mask in provider._instances(provider._call(small, "tree"), small.size):
                trees |= mask
            reference["foliage"] = trees
    except RuntimeError as exc:
        # SAM is required for every detection; V3 failing is handled below, not here.
        raise HTTPException(503, str(exc))
    lap(f"segmentation ({len(detections)} instances, {settings.edge_extractor}"
        + (f"/{settings.partition_engine}" if partitioned else "") + ")")

    # The roof planes at working size, as SAM returned them: what the target-house tap stores.
    working_roofs = [d.mask for d in detections if d.kind == "roof"]
    if reference is not None:
        # the reference geometry runs on the masks at working size, as they came back from SAM
        reference["working_masks"] = list(working_roofs)
    for detection in detections:
        detection.mask = detect.upscale_mask(detection.mask, image.size)
    # Context for the crop only: never a region, never a blocker.
    skies = [d.mask for d in detections if d.kind == "sky"]
    sky = np.logical_or.reduce(skies) if skies else None
    detections = [d for d in detections if d.kind != "sky"]

    frame = (image.height, image.width)
    blockers = occlusion.blocker_union([d.mask for d in detections if _blocks(d.kind)], frame)
    solid = occlusion.blocker_union([d.mask for d in detections if d.kind in occlusion.SOLID_KINDS], frame)
    fresh: list[RegionRow] = []

    roofs = [d for d in detections if d.kind == "roof"]
    roof_masks = [d.mask for d in roofs]
    scores = [d.score for d in roofs]
    details: dict = {}
    target_info: dict | None = None
    extra: dict = {}
    detection_run: dict | None = None
    if roof_wanted and (roof_masks or use_v3):
        # The target house (m0): decides the V3 crop always, and which lines stand only with
        # TARGET_HOUSE_ONLY, exactly as before.
        keep = target.select(roof_masks, image) if roof_masks and (settings.target_house_only or use_v3) else None
        house = keep if settings.target_house_only else None
        vp_region = target.evidence_region(roof_masks, house) if house is not None else None
        if house is not None:
            # Every plane is stored with its keep flag, so a tap in the editor can move one in or
            # out of the house without another detection call.
            store.write_planes(photo.id, working_roofs)
            target_info = {
                # Scores unrounded: the partition ranks planes by them, and a tap must redraw the
                # lines from exactly what detection used.
                "planes": [{"outline": outline, "keep": keep_it, "score": float(score)}
                           for outline, keep_it, score in zip(target.outlines(roof_masks), house, scores)],
                "sides": parsed.sides,
                "edges": parsed.edges,
            }
        crop = None
        v3 = None
        if use_v3:
            crop, top = rooflines.target_crop(roof_masks, keep, image.size, settings.v3_crop_padding,
                                              sky=sky, sky_max_rise=settings.v3_crop_sky_max_rise)
            # "target_house": padded box; "target_house_sky": its top raised to where the sky begins.
            basis = "target_house_sky" if top == "sky" else "target_house"
            if crop is None:
                # SAM found no roof plane: V3 still gets the photo rather than nothing.
                crop, basis = rooflines.CropBox(0, 0, image.width, image.height), "whole_photo"
            v3 = fusion.run_v3(image, crop, settings, basis)
            if settings.v3_missed_planes and v3.usable:
                # Second calls for target-house planes the first answer left without lines.
                v3.meta["missed_planes"] = fusion.run_missed_planes(image, v3, roof_masks, keep, blockers, sky, settings)
            detection_run = {**v3.meta, "detector": "v3_sol", "status": v3.status, "error": v3.error,
                             "fallback_used": not v3.usable, "side_review_required": bool(parsed.sides)}
            extra_costs = [call.meta.get("cost_usd") for call in v3.extra]
            if extra_costs:
                # This run's bill across every V3 call; unknown when any call's cost is unknown.
                costs = [v3.meta.get("cost_usd"), *extra_costs]
                detection_run["total_cost_usd"] = None if any(c is None for c in costs) else round(sum(costs), 6)
            lap(f"v3 {v3.status} ({len(v3.raw_lines)} lines, cache {v3.meta.get('cache')}, crop {crop.as_list()}"
                + (f", {len(v3.extra)} second calls" if v3.extra else "") + ")")

        def partition_lines(wanted: intent.Intent):
            # Only called when partition draws the lines: configured alone, or V3 failed.
            if not roof_masks:
                return [], [], {}
            spans, unrequested, found = compute_roof_lines(
                roof_masks, scores, blockers, solid, wanted, image, reference=reference, vp_region=vp_region,
            )
            if house is not None:
                spans, unrequested = target.on_house(spans, roof_masks, house), target.on_house(unrequested, roof_masks, house)
            return spans, unrequested, found

        result = fusion.unify(
            image=image, settings=settings, parsed=parsed, roof_masks=roof_masks,
            keep=keep if use_v3 else house, blockers=blockers, partition_lines=partition_lines, v3=v3,
            vp_region=vp_region, crop=crop,
        )
        details = result.details
        fresh += roof_rows(result, parsed)
        extra.update(_roofline_extra(result, image, store, photo))
        lap(f"roof lines ({sum(result.requested)} lit, {len(result.lines) - sum(result.requested)} kept for review"
            + (f", {sum(house)} of {len(house)} planes on the target house" if house is not None else "") + ")")

    # Anything detected that the instruction did not ask for is here as evidence of what
    # stands in front of the house, not as something to light.
    targets = set(selected or detect.DEFAULT_KINDS)
    areas = [d for d in detections if d.kind != "roof"]
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for item in areas:
        totals[item.kind] = totals.get(item.kind, 0) + 1

    for item in areas:
        index = counts.get(item.kind, 0)
        counts[item.kind] = index + 1
        region_id = uuid.uuid4()
        fresh.append(
            RegionRow(
                id=region_id,
                kind=item.kind,
                role="target" if item.kind in targets else "blocker",
                label=detect.label_for(item.kind, index, totals[item.kind]),
                geometry="area",
                source="auto",
                color=detect.COLORS.get(item.kind, detect.COLORS["custom"]),
                score=round(item.score, 3),
                visible=True,
                side=None,
                edge=None,
                polyline=None,
                mask_url=store.write_mask(photo.id, region_id, item.mask),
            )
        )

    if details.get("pitch") or details.get("planes") is not None:
        # Pitch and plane polygons ride with the interpretation; there is no plane entity to hang them on.
        extra.update({"pitch": {str(k): v for k, v in (details.get("pitch") or {}).items()}, "planes": details.get("planes")})
    if target_info is not None:
        extra["target"] = target_info
    if roof_wanted:
        # V3's billing and latency record for this run; None when V3 is not a source.
        extra["detection_run"] = detection_run
    if extra:
        store.update_photo(photo, interpreted={**(photo.interpreted or {}), **extra})
    store.replace_auto_regions(photo, fresh)
    result = [region_dict(region) for region in store.refresh_hidden(photo)]
    lap("store")
    return result


@router.post("/target", response_model=PhotoDetail)
def tap_target_plane(photo_id: str, body: TargetTap, store: Store = Depends(get_store)):
    """Move the roof plane under a tap into or out of the house being quoted.

    The roof lines are redrawn from the planes stored at detection, through the same unified
    geometry and with the instruction's side and edge filter from then. No detection call is made:
    no SAM call, and no V3 call — V3's stored answer for the original crop is used again, its
    lines on planes now left out are filtered away. Area regions and lines someone drew are
    untouched; detected roof lines are redrawn, so an edit to one of them is undone by a tap.
    Lighting already confirmed carries over to the redrawn lines.
    """
    settings = get_settings()
    if not settings.target_house_only:
        # The flag switches the whole feature off, taps included.
        raise HTTPException(409, "TARGET_HOUSE_ONLY is off. Turn it on and detect again to pick the house's planes.")
    photo = store.photo(photo_id)
    info = (photo.interpreted or {}).get("target")
    if not info or not info.get("planes"):
        raise HTTPException(409, "This photo has no stored roof planes. Detect it with TARGET_HOUSE_ONLY on.")
    if settings.edge_extractor == "partition" and settings.partition_engine == "reference":
        raise HTTPException(409, "Tapping planes redraws lines with this repo's engine: set PARTITION_ENGINE=port.")
    working = store.read_planes(photo.id, len(info["planes"]))
    if working is None:
        raise HTTPException(409, "This photo's stored roof planes are missing. Detect it again.")
    masks = [detect.upscale_mask(mask, (photo.width, photo.height)) for mask in working]
    index = target.plane_at(masks, body.point)
    if index is None:
        raise HTTPException(404, "No roof plane there. Tap on a roof.")
    keep = [bool(plane["keep"]) for plane in info["planes"]]
    moved = not keep[index]
    for plane in target.copies_of(masks, index):
        keep[plane] = moved
    if not any(keep):
        raise HTTPException(400, "The house needs at least one roof plane.")

    regions = store.regions(photo)
    blockers = store.area_union(photo, regions, occlusion.BLOCKER_KINDS)
    solid = store.area_union(photo, regions, occlusion.SOLID_KINDS)
    image = store.load_image(photo)
    stored_v3 = ((photo.interpreted or {}).get("roofline") or {}).get("v3")
    store.release()
    parsed = intent.Intent(sides=list(info.get("sides") or []), edges=list(info.get("edges") or []))
    scores = [float(plane.get("score", 1.0)) for plane in info["planes"]]
    vp_region = target.evidence_region(masks, keep)
    v3 = fusion.V3Outcome.from_stored(stored_v3) if "v3" in settings.roofline_sources else None

    def partition_lines(wanted: intent.Intent):
        spans, unrequested, found = compute_roof_lines(masks, scores, blockers, solid, wanted, image, vp_region=vp_region)
        return target.on_house(spans, masks, keep), target.on_house(unrequested, masks, keep), found

    result = fusion.unify(
        image=image, settings=settings, parsed=parsed, roof_masks=masks, keep=keep, blockers=blockers,
        partition_lines=partition_lines, v3=v3, vp_region=vp_region, crop=v3.crop if v3 else None,
    )
    extra = _roofline_extra(result, image, store, photo)
    extra["roofline"] = {**extra["roofline"], "v3_reused": v3 is not None}
    changes: dict = {"target": {**info, "planes": [{**plane, "keep": keep_it} for plane, keep_it in zip(info["planes"], keep)]},
                     **extra}
    if result.details.get("pitch") is not None:
        changes["pitch"] = {str(k): v for k, v in result.details["pitch"].items()}
    if store.replace_roof_lines(photo, roof_rows(result, parsed), changes, info) is None:
        raise HTTPException(409, "This photo was detected or tapped again while its lines were redrawn. Tap again.")
    _relight(store, photo)
    return {**photo_dict(photo), "regions": [region_dict(region) for region in store.refresh_hidden(photo)]}


# Find in Box: the box the user drew is padded this much for context, and must be this big.
FIND_PADDING = 0.05
FIND_MIN_BOX = 16


@router.post("/find-lines", response_model=list[Region])
def find_lines_in_box(photo_id: str, body: FindLinesRequest, store: Store = Depends(get_store)):
    """Find in Box: detect roof lines inside a box the user drew, where detection missed some.

    Semi-automatic: the user says where, the system detects. V3 (if a source) reads the padded
    box, cached with the box in its key, and its lines go through the same pipeline as /detect.
    Only when partition is the source, or V3 failed, is SAM asked for roof planes in the crop and
    partition's lines used instead. Only lines whose middle lies in
    the box and that do not repeat a line already on the photo are added, as new detected lines.
    Nothing existing is moved, merged, relabelled or deleted, and lines someone drew never are.
    """
    photo = store.photo(photo_id)
    x0, y0, x1, y1 = body.box
    x0, x1 = sorted((max(0, min(x0, photo.width)), max(0, min(x1, photo.width))))
    y0, y1 = sorted((max(0, min(y0, photo.height)), max(0, min(y1, photo.height))))
    if x1 - x0 < FIND_MIN_BOX or y1 - y0 < FIND_MIN_BOX:
        raise HTTPException(400, f"Draw a larger box: at least {FIND_MIN_BOX} px each way.")
    settings = get_settings()
    image = store.load_image(photo)
    regions = store.regions(photo)
    blockers = store.area_union(photo, regions, occlusion.BLOCKER_KINDS)
    solid = store.area_union(photo, regions, occlusion.SOLID_KINDS)
    existing = [
        rooflines.RoofLine(points=[list(map(float, p)) for p in r.polyline], source=list(r.sources or [r.source]),
                           line_type=r.edge, side=r.side, source_id=str(r.id))
        for r in regions if r.geometry == "line" and r.kind == "roof" and r.polyline and len(r.polyline) >= 2
    ]
    store.release()

    crop = rooflines.padded_box(x0, y0, x1, y1, FIND_PADDING, image.size)
    v3 = fusion.run_v3(image, crop, settings, "find_in_box") if "v3" in settings.roofline_sources else None
    roof_masks: list[np.ndarray] = []
    scores: list[float] = []
    if not (v3 and v3.usable) and "partition" in settings.roofline_sources:
        piece = image.crop(tuple(crop.as_list()))
        small, _ = detect.working_copy(piece, settings.detect_max_edge)
        try:
            found = detect.run(small, get_provider(), ["roof"], False)
        except RuntimeError as exc:
            if not (v3 and v3.usable):
                raise HTTPException(503, str(exc))
            found = []
        for item in found:
            full = np.zeros((image.height, image.width), bool)
            full[crop.y0:crop.y1, crop.x0:crop.x1] = detect.upscale_mask(item.mask, (crop.width, crop.height))
            roof_masks.append(full)
            scores.append(item.score)

    def partition_lines(wanted: intent.Intent):
        if not roof_masks:
            return [], [], {}
        # This repo's engine on the box's planes: the reference engine needs a whole-photo working copy.
        return compute_roof_lines(roof_masks, scores, blockers, solid, wanted, image)

    result = fusion.unify(
        image=image, settings=settings, parsed=intent.Intent(), roof_masks=roof_masks, keep=None,
        blockers=blockers, partition_lines=partition_lines, v3=v3, crop=crop,
    )
    inside = []
    for line in result.lines:
        middle = (np.asarray(line.points[0], float) + np.asarray(line.points[-1], float)) / 2
        if x0 <= middle[0] <= x1 and y0 <= middle[1] <= y1:
            inside.append(line)
    context = fusion.Context(image, settings, roof_masks, None, blockers)
    new = fusion.dedupe_against(inside, existing, context)
    found_result = fusion.Result(new, [True] * len(new), result.inferred, {}, result.report, None)
    rows = roof_rows(found_result, intent.Intent(), found_in_box=True)

    photo = store.photo(photo_id)
    added = [store.add_region(photo, row) for row in rows]
    report = {**result.report, "box": [x0, y0, x1, y1], "added": len(added),
              "duplicates_skipped": len(inside) - len(new)}
    report.pop("v3", None)
    store.update_photo(photo, interpreted={**(photo.interpreted or {}), "roofline_find": report})
    ids = {region.id for region in added}
    return [region_dict(region) for region in store.refresh_hidden(photo) if region.id in ids]


def _relight(store: Store, photo) -> None:
    """Carry confirmed lighting over to redrawn roof lines. The old lines' rows went with them,
    so each confirmed request lights what it matches now, the way put_design matched it."""
    design = store.design(photo)
    if design is None:
        return
    regions = store.regions(photo)
    rows: dict = {}
    confirmed = []
    for request in design.requests or []:
        matched = [region for region in regions if lighting.matches(request, region)]
        for region in matched:
            # A later request naming the same region wins, as in put_design.
            rows[region.id] = request["settings"]
        confirmed.append({**request, "region_ids": [str(region.id) for region in matched]})
    store.set_design(photo, confirmed, rows)


@router.post("/regions", response_model=Region)
def create_region(photo_id: str, body: CreateRegionRequest, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    if body.geometry == "line" and not body.polyline:
        raise HTTPException(400, "A line region needs a polyline")

    polyline, side = body.polyline, body.side
    # Eave, rake and ridge only mean something on a roof; a driveway or fence line carries no edge.
    edge = None
    if body.geometry == "line" and body.kind == "roof":
        # A hand-drawn roofline joins the lines already there and gets a suggested type and side,
        # shown with the type picker (services/drawn.py). Without a camera there is no level test.
        roofs = [r for r in store.regions(photo) if r.geometry == "line" and r.kind == "roof" and r.polyline]
        size = (photo.width, photo.height)
        polyline = [[round(v) for v in point]
                    for point in drawn.snap(body.polyline, [r.polyline for r in roofs], size)]
        try:
            found = photos_router._camera(store, photo)
        except Exception:  # noqa: BLE001 - a suggestion never blocks drawing a line
            log.exception("no camera for a drawn line on %s", photo.id)
            found = None
        edge, suggested_side = drawn.suggest(polyline, [(r.polyline, r.edge, r.side) for r in roofs], found, size)
        side = body.side or suggested_side

    region = RegionRow(
        id=uuid.uuid4(),
        kind=body.kind,
        label=body.label or detect.LABELS.get(body.kind, "Region"),
        geometry=body.geometry,
        source="manual",
        color=body.color or (detect.EDGE_COLORS.get(edge) if edge else None)
        or detect.COLORS.get(body.kind, detect.COLORS["custom"]),
        score=1.0,
        visible=True,
        side=side,
        edge=edge,
        polyline=polyline,
        mask_url=None,
    )
    if body.geometry == "area":
        blank = np.zeros((photo.height, photo.width), dtype=bool)
        region.mask_url = store.write_mask(photo.id, region.id, blank)

    region = store.add_region(photo, region)
    if region.geometry == "line" or _blocks(region.kind):
        store.refresh_hidden(photo)
    if region.geometry == "line" and region.kind == "roof":
        # Confirmed lighting reaches the new line the way it reached the detected ones.
        _relight(store, photo)
    return region_dict(region)


@router.patch("/regions/{region_id}", response_model=Region)
def patch_region(
    photo_id: str, region_id: str, body: RegionPatch, store: Store = Depends(get_store)
):
    photo = store.photo(photo_id)
    region = store.region(photo, region_id, lock=True)

    updates = body.model_dump(exclude_unset=True)
    if "polyline" in updates and region.geometry != "line":
        raise HTTPException(400, "Only line regions carry a polyline")
    if "kind" in updates:
        region.color = detect.COLORS.get(updates["kind"], detect.COLORS["custom"])
    for name, value in updates.items():
        setattr(region, name, value)

    store.save()
    if region.geometry == "line" and updates.keys() & {"edge", "side", "kind", "visible"}:
        # A line given its type (or side) now matches the confirmed requests that name it.
        _relight(store, photo)
    if "kind" in updates and _blocks(updates["kind"]) and region.geometry == "area":
        # Something just became an object in front of the house: mend the lines behind it.
        store.rejoin_behind_blockers(photo)
    elif "polyline" in updates or "kind" in updates:
        # A dragged line must never be pulled back to where it was, so only the hidden
        # stretches are recomputed here.
        store.refresh_hidden(photo)
    return region_dict(region)


@router.put("/regions/{region_id}/mask", response_model=Region)
def put_mask(
    photo_id: str,
    region_id: str,
    file: UploadFile = File(...),
    store: Store = Depends(get_store),
):
    photo = store.photo(photo_id)
    region = store.region(photo, region_id, lock=True)
    if region.geometry != "area":
        raise HTTPException(400, "Only area regions carry a mask")

    region.mask_url = store.write_mask_bytes(photo.id, region.id, file.file.read())
    region.source = "manual"
    store.save()
    if _blocks(region.kind):
        # Painting the palm the model missed is enough: the roofline behind it joins back up.
        store.rejoin_behind_blockers(photo)
    return region_dict(region)


@router.post("/regions/refine", response_model=Region)
def refine(photo_id: str, body: RefineRequest, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    # Reject a bad target before the model call — every call to the hosted model is billed.
    if body.region_id and store.region(photo, body.region_id).geometry != "area":
        raise HTTPException(400, "Only area regions can be refined")

    image = store.load_image(photo)
    small, scale = detect.working_copy(image, get_settings().detect_max_edge)

    if body.box is not None:
        x0, y0, x1, y1 = (round(v / scale) for v in body.box)
        if x1 <= x0 or y1 <= y0:
            raise HTTPException(400, "Box must have positive width and height")
        prompt = lambda provider: provider.segment_box(small, (x0, y0, x1, y1))  # noqa: E731
    else:
        point = tuple(round(v / scale) for v in body.point)
        prompt = lambda provider: provider.segment_point(small, point)  # noqa: E731

    store.release()
    try:
        mask = detect.upscale_mask(
            detect.clean(prompt(get_provider()), small.width * small.height), image.size
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))

    if body.region_id:
        region = store.region(photo, body.region_id, lock=True)
        existing = store.read_mask(photo.id, region.id)
        if existing is not None:
            mask = np.logical_or(existing, mask)
        region.mask_url = store.write_mask(photo.id, region.id, mask)
        store.save()
    else:
        region = RegionRow(
            id=uuid.uuid4(),
            kind=body.kind,
            label=detect.LABELS.get(body.kind, "Region"),
            geometry="area",
            source="auto",
            color=detect.COLORS.get(body.kind, detect.COLORS["custom"]),
            score=0.5,
            visible=True,
            side=None,
            edge=None,
            polyline=None,
        )
        region.mask_url = store.write_mask(photo.id, region.id, mask)
        region = store.add_region(photo, region)

    if _blocks(region.kind):
        store.rejoin_behind_blockers(photo)
    return region_dict(region)


@router.delete("/regions/{region_id}", status_code=204)
def delete_region(photo_id: str, region_id: str, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    if _blocks(store.delete_region(photo, region_id).kind):
        store.refresh_hidden(photo)
