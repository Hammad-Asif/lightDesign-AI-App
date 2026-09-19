import logging
import time
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.models import Region as RegionRow
from app.providers.base import get_provider
from app.schemas import CreateRegionRequest, DetectRequest, RefineRequest, Region, RegionPatch
from app.services import detect, intent, lighting, occlusion, partition, partition_reference, roofline, snap
from app.storage import Store, get_store, region_dict

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
    interior: bool = False,
    partitioned: bool = False,
    image=None,
    scores: list[float] | None = None,
    details: dict | None = None,
    junctions: bool = True,
    constrained: bool = True,
    unrequested: list | None = None,
    reference: dict | None = None,
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
            roof_masks, width, blockers, image=image, scores=scores, junctions=junctions, constrained=constrained
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
        detections = detect.run(small, provider, selected, settings.detect_blockers)
        if use_reference and settings.partition_tree_pass and hasattr(provider, "_call"):
            trees = np.zeros((small.height, small.width), bool)
            for _, mask in provider._instances(provider._call(small, "tree"), small.size):
                trees |= mask
            reference["foliage"] = trees
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    lap(f"segmentation ({len(detections)} instances, {settings.edge_extractor}"
        + (f"/{settings.partition_engine}" if partitioned else "") + ")")

    if reference is not None:
        # the reference geometry runs on the masks at working size, as they came back from SAM
        reference["working_masks"] = [d.mask for d in detections if d.kind == "roof"]
    for detection in detections:
        detection.mask = detect.upscale_mask(detection.mask, image.size)

    frame = (image.height, image.width)
    blockers = occlusion.blocker_union([d.mask for d in detections if _blocks(d.kind)], frame)
    fresh: list[RegionRow] = []

    roofs = [d for d in detections if d.kind == "roof"]
    roof_masks = [d.mask for d in roofs]
    details: dict = {}
    unrequested: list[tuple] = []
    if roof_masks:
        # The reference geometry is final after its relaxation; snapping on top is opt-in there.
        snapping = settings.snap_to_edges and (not partitioned or settings.partition_snap)
        spans = roof_lines(
            roof_masks,
            blockers,
            parsed,
            settings.edge_extractor == "hybrid",
            snap.field(image) if snapping or settings.interior_hips else None,
            settings.interior_hips,
            partitioned,
            image=image,
            scores=[d.score for d in roofs],
            details=details,
            junctions=settings.partition_junctions,
            constrained=settings.partition_constrained_snap,
            unrequested=unrequested,
            reference=reference,
        )
        lap(f"roof lines ({len(spans)} spans, {len(unrequested)} detected but not requested)")
        inferred = details.get("inferred", set())
        for index, (polyline, side, edge) in enumerate(spans):
            fresh.append(
                RegionRow(
                    kind="roof",
                    label=f"{EDGE_LABEL[edge]} — {SIDE_LABEL[side]} {index + 1}",
                    geometry="line",
                    # A connector drawn across an occluded stretch from no pixels is kept as
                    # data, marked so the editor can show it for what it is.
                    source="inferred" if _key((polyline,)) in inferred else "auto",
                    color=detect.edge_color(edge),
                    score=0.5,
                    visible=True,
                    side=side,
                    edge=edge,
                    polyline=polyline,
                    mask_url=None,
                )
            )

    for index, (polyline, side, edge) in enumerate(unrequested):
        # Detected by the partition, not asked for by the instruction: kept, hidden, so the
        # editor shows the same roof structure the reference method draws and one click lights it.
        fresh.append(
            RegionRow(
                kind="roof",
                label=f"{EDGE_LABEL.get(edge, edge)} — {SIDE_LABEL.get(side, side)} {index + 1} (not lit)",
                geometry="line",
                source="auto",
                color=detect.edge_color(edge),
                score=0.5,
                visible=False,
                side=side,
                edge=edge,
                polyline=polyline,
                mask_url=None,
            )
        )

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
        store.update_photo(
            photo,
            interpreted={**(photo.interpreted or {}), "pitch": {str(k): v for k, v in (details.get("pitch") or {}).items()},
                         "planes": details.get("planes")},
        )
    store.replace_auto_regions(photo, fresh)
    result = [region_dict(region) for region in store.refresh_hidden(photo)]
    lap("store")
    return result


@router.post("/regions", response_model=Region)
def create_region(photo_id: str, body: CreateRegionRequest, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    if body.geometry == "line" and not body.polyline:
        raise HTTPException(400, "A line region needs a polyline")

    region = RegionRow(
        id=uuid.uuid4(),
        kind=body.kind,
        label=body.label or detect.LABELS.get(body.kind, "Region"),
        geometry=body.geometry,
        source="manual",
        color=body.color or detect.COLORS.get(body.kind, detect.COLORS["custom"]),
        score=1.0,
        visible=True,
        side=body.side,
        # Eave, rake and ridge only mean something on a roof. A hand-drawn roofline is not known
        # to be any of them, so say so; a driveway or fence line carries no edge type at all.
        edge="unknown" if body.geometry == "line" and body.kind == "roof" else None,
        polyline=body.polyline,
        mask_url=None,
    )
    if body.geometry == "area":
        blank = np.zeros((photo.height, photo.width), dtype=bool)
        region.mask_url = store.write_mask(photo.id, region.id, blank)

    region = store.add_region(photo, region)
    if region.geometry == "line" or _blocks(region.kind):
        store.refresh_hidden(photo)
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
    if "polyline" in updates or "kind" in updates:
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
        store.refresh_hidden(photo)
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
        store.refresh_hidden(photo)
    return region_dict(region)


@router.delete("/regions/{region_id}", status_code=204)
def delete_region(photo_id: str, region_id: str, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    if _blocks(store.delete_region(photo, region_id).kind):
        store.refresh_hidden(photo)
