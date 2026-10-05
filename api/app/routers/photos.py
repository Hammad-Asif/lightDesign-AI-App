import uuid

import numpy as np

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models import Photo as PhotoRow
from app.models import Region as RegionRow
from app.schemas import (DesignOut, DesignPut, InterpretRequest, ModeRequest, Photo, PhotoDetail, ScaleCheckPut,
                         ScaleOverride, ScalePut, WreathRequest)
from app.services import camera, detect, fusion, lighting, patterns, placement, scale, scale_debug
from app.storage import Store, get_store, photo_dict, region_dict

router = APIRouter(prefix="/photos", tags=["photos"])

ALLOWED = {"image/jpeg", "image/png", "image/webp"}


def _detail(store: Store, photo: PhotoRow) -> dict:
    return {**photo_dict(photo), "regions": [region_dict(r) for r in store.regions(photo)]}


@router.get("", response_model=list[Photo])
def list_photos(store: Store = Depends(get_store)):
    return [photo_dict(photo) for photo in store.list_photos()]


@router.post("", response_model=Photo)
def upload(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    store: Store = Depends(get_store),
):
    settings = get_settings()
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Upload a JPG, PNG or WEBP image")
    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB")
    try:
        photo = store.create_photo(data, prompt.strip(), settings.max_image_edge)
    except OSError:
        raise HTTPException(400, "Could not read that image file")
    # Read the instruction straight away: what it resolved to is shown before any detection.
    return photo_dict(_interpret(store, photo))


def _interpret(store: Store, photo: PhotoRow) -> PhotoRow:
    read = lighting.interpretation(photo.prompt, store.products())
    return store.update_photo(photo, interpreted={**(photo.interpreted or {}), **read})


@router.get("/{photo_id}", response_model=PhotoDetail)
def detail(photo_id: str, store: Store = Depends(get_store)):
    return _detail(store, store.photo(photo_id))


@router.put("/{photo_id}/mode", response_model=Photo)
def set_mode(photo_id: str, body: ModeRequest, store: Store = Depends(get_store)):
    return photo_dict(store.update_photo(store.photo(photo_id), mode=body.mode))


@router.post("/{photo_id}/approve", response_model=PhotoDetail)
def approve(photo_id: str, store: Store = Depends(get_store)):
    photo = store.photo(photo_id)
    if not store.regions(photo):
        raise HTTPException(400, "Approve requires at least one region")
    return _detail(store, store.update_photo(photo, approved=True))


@router.post("/{photo_id}/reopen", response_model=PhotoDetail)
def reopen(photo_id: str, store: Store = Depends(get_store)):
    return _detail(store, store.update_photo(store.photo(photo_id), approved=False))


@router.post("/{photo_id}/interpret", response_model=Photo)
def interpret(photo_id: str, body: InterpretRequest, store: Store = Depends(get_store)):
    """Re-read the instruction — a changed one, or the same one against a changed catalogue —
    without detecting anything. Detection is billed; reading is not."""
    photo = store.photo(photo_id)
    if body.prompt is not None:
        photo = store.update_photo(photo, prompt=body.prompt.strip())
    return photo_dict(_interpret(store, photo))


def _design_out(store: Store, design) -> dict:
    return {
        "id": str(design.id),
        "name": design.name,
        "requests": design.requests or [],
        "regions": [{"region_id": str(row.region_id), "settings": row.settings}
                    for row in store.design_regions(design)],
    }


@router.get("/{photo_id}/design", response_model=DesignOut | None)
def get_design(photo_id: str, store: Store = Depends(get_store)):
    """The confirmed lighting, or null while nothing has been confirmed — an ordinary state."""
    design = store.design(store.photo(photo_id))
    return _design_out(store, design) if design is not None else None


@router.put("/{photo_id}/design", response_model=DesignOut)
def put_design(photo_id: str, body: DesignPut, store: Store = Depends(get_store)):
    """Confirm the lighting: each request's settings go onto every region it lights. Nothing
    renders from an instruction directly — only from what was confirmed here."""
    photo = store.photo(photo_id)
    regions = store.regions(photo)
    products = {str(product.id): product for product in store.products()}
    rows: dict[uuid.UUID, dict] = {}
    confirmed: list[dict] = []
    problems: list[str] = []
    for index, request in enumerate(body.requests, start=1):
        product = products.get(str(request.settings.product_id))
        if product is None:
            problems.append(f"Request {index}: that product is not in the catalogue")
            continue
        asked = request.model_dump(mode="json")
        matched = [region for region in regions if lighting.matches(asked, region)]
        wrong = {problem for region in matched for problem in patterns.problems(request.settings, product, region.kind)}
        if wrong:
            problems += [f"Request {index}: {problem}" for problem in sorted(wrong)]
            continue
        settings = request.settings.model_dump(mode="json")
        for region in matched:
            # A later request naming the same region wins, as the later sentence would.
            rows[region.id] = settings
        confirmed.append({**asked, "product_name": product.name,
                          "region_ids": [str(region.id) for region in matched]})
    if problems:
        raise HTTPException(422, problems)
    return _design_out(store, store.set_design(photo, confirmed, rows))


def _camera(store: Store, photo: PhotoRow) -> camera.Camera:
    """The photo's camera as detection stored it. A photo detected before cameras were stored
    gets one from its whole-photo vanishing points: read from the image, no billed call."""
    stored = (photo.interpreted or {}).get("camera")
    if stored:
        # Under today's focal rules: a camera stored before a rule changed is tested again.
        return camera.rechecked(camera.Camera.from_dict(stored))
    image = store.load_image(photo)
    found = camera.build(fusion.Context(image, get_settings(), [], None, None).scene, image.size, photo.focal_35mm)
    store.update_photo(photo, interpreted={**(photo.interpreted or {}), "camera": found.to_dict()})
    return found


@router.put("/{photo_id}/scale", response_model=Photo)
def put_scale(photo_id: str, body: ScalePut, store: Store = Depends(get_store)):
    """The salesperson's one known length: two points tapped on a feature and its real size, or
    a line (an eave) and its real length. It sets the scale of the wall it lies on; the other
    walls take theirs through the corners they share with it. Null clears it."""
    photo = store.photo(photo_id)
    if body.measurement is None:
        rest = {k: v for k, v in (photo.scale or {}).items() if k != "measurement"}
        return photo_dict(store.update_photo(photo, scale=rest or None))
    measurement = body.measurement
    found = _camera(store, photo)
    # A side the salesperson chose stands. Otherwise the solve decides it from the photo's
    # structure (services/walls.py), every time: what is stored here is only a hint.
    given = measurement.facade is not None
    facade = measurement.facade
    # The sides of the cameras the solve may use instead (when the gables' rakes reject this one).
    sides = list(dict.fromkeys([*found.facades, *(f for _, other in scale.alternatives(found) for f in other.facades)]))
    if given and sides and facade not in sides:
        raise HTTPException(400, f"No {facade} wall was found in this photo's perspective. Measure on the "
                                 + " or ".join(sides) + " side.")
    if measurement.region_id is not None:
        region = store.region(photo, measurement.region_id)
        if region.geometry != "line" or not region.polyline or len(region.polyline) < 2:
            raise HTTPException(400, "Measure along a line, such as an eave")
        facade = facade or region.side
    else:
        if any(not (0 <= x < photo.width and 0 <= y < photo.height) for x, y in measurement.points):
            raise HTTPException(400, "Both points must be on the photo")
        facade = facade or found.facade_of(*measurement.points, get_settings().roofline_vp_deg)
    record = {**measurement.model_dump(exclude_none=True), "facade": facade, "facade_given": given}
    if measurement.region_id is not None:
        # The line as it was measured: the scale follows the line, so an edit to it later is said.
        record["line_points"] = region.polyline
    return photo_dict(store.update_photo(photo, scale={**(photo.scale or {}), "measurement": record}))


@router.put("/{photo_id}/scale/check", response_model=Photo)
def put_scale_check(photo_id: str, body: ScaleCheckPut, store: Store = Depends(get_store)):
    """A second known length on a line: it changes nothing, and the solve says how far the photo's
    length for that line is from it (GET /geometry: reference_check). Null clears it."""
    photo = store.photo(photo_id)
    rest = {k: v for k, v in (photo.scale or {}).items() if k != "check"}
    if body.check is None:
        return photo_dict(store.update_photo(photo, scale=rest or None))
    region = store.region(photo, body.check.region_id)
    if region.geometry != "line" or not region.polyline or len(region.polyline) < 2:
        raise HTTPException(400, "Check against a line, such as an eave")
    return photo_dict(store.update_photo(photo, scale={**rest, "check": body.check.model_dump()}))


@router.put("/{photo_id}/scale/overrides/{region_id}", response_model=Photo)
def put_override(photo_id: str, region_id: str, body: ScaleOverride | None = None, store: Store = Depends(get_store)):
    """The salesperson's corrections to one region's placement (see ScaleOverride): a line's
    midpoint and length, an area's count or density, a trunk's strand length, a wreath's size.
    Null clears them. A wreath's new size redraws its disc."""
    photo = store.photo(photo_id)
    region = store.region(photo, region_id)
    overrides = dict((photo.scale or {}).get("overrides") or {})
    if body is None:
        overrides.pop(region_id, None)
    else:
        record = body.model_dump(exclude_none=True)
        line_keys, area_keys, wreath_keys = {"mid", "length_in"}, {"count", "density", "strand_ft"}, {"at", "size_in"}
        allowed = (line_keys if region.geometry == "line"
                   else wreath_keys if region.kind == "wreath" else area_keys)
        if set(record) - allowed:
            raise HTTPException(400, f"A {region.kind.replace('_', ' ')} cannot take " + ", ".join(sorted(set(record) - allowed)))
        if "mid" in record and (not region.polyline or len(region.polyline) < 2):
            raise HTTPException(400, "Only a line has a midpoint to set")
        overrides[region_id] = record
        if region.kind == "wreath" and "size_in" in record:
            at = record.get("at") or (overrides.get(region_id) or {}).get("at")
            _draw_wreath(store, photo, region, at, record["size_in"])
    rest = {k: v for k, v in (photo.scale or {}).items() if k != "overrides"}
    return photo_dict(store.update_photo(photo, scale={**rest, **({"overrides": overrides} if overrides else {})} or None))


def _wreath_scale(store: Store, photo: PhotoRow, point) -> float:
    found, solved = _solve(store, photo, store.regions(photo))
    lines = {str(r.id): r.polyline for r in store.regions(photo) if r.geometry == "line" and r.polyline}
    local = placement.wall_scale(found, solved, lines, point)
    if local is None:
        raise HTTPException(409, "Measure the house first, so the wreath can be drawn at its real size")
    return local[0]


def _draw_wreath(store: Store, photo: PhotoRow, region: RegionRow, at, size_in: float) -> None:
    """The wreath's disc: its real diameter at the local scale, centred where it was tapped."""
    if not at:
        mask = store.read_mask(photo.id, region.id)
        at = [int(v) for v in np.argwhere(mask).mean(axis=0)[::-1]] if mask is not None and mask.any() else None
    if not at:
        raise HTTPException(400, "This wreath has no position")
    radius = size_in / 2 * _wreath_scale(store, photo, at)
    ys, xs = np.ogrid[:photo.height, :photo.width]
    region.mask_url = store.write_mask(photo.id, region.id, (xs - at[0]) ** 2 + (ys - at[1]) ** 2 <= radius ** 2)
    region.label = f"Wreath, {size_in} in"
    store.save()


@router.post("/{photo_id}/wreaths", response_model=PhotoDetail)
def add_wreath(photo_id: str, body: WreathRequest, store: Store = Depends(get_store)):
    """A wreath where the salesperson tapped (Lukas: no automatic placement), drawn at its real
    size by the local scale. Kept like anything drawn by hand: detecting again never removes it."""
    photo = store.photo(photo_id)
    x, y = body.point
    if not (0 <= x < photo.width and 0 <= y < photo.height):
        raise HTTPException(400, "Tap on the photo")
    _wreath_scale(store, photo, body.point)      # refuse before creating anything
    region = RegionRow(id=uuid.uuid4(), kind="wreath", role="target", label=f"Wreath, {body.size_in} in",
                       geometry="area", source="manual", color=detect.COLORS.get("wreath", detect.COLORS["custom"]),
                       score=1.0, visible=True, side=None, edge=None, polyline=None, mask_url=None)
    region = store.add_region(photo, region)
    _draw_wreath(store, photo, region, body.point, body.size_in)
    overrides = dict((photo.scale or {}).get("overrides") or {})
    overrides[str(region.id)] = {"at": list(body.point), "size_in": body.size_in}
    store.update_photo(photo, scale={**(photo.scale or {}), "overrides": overrides})
    return _detail(store, photo)


def roof_lines(regions, topology: dict | None = None) -> list[scale.Line]:
    """The photo's roof lines as the scale solver reads them."""
    return [scale.Line(str(r.id), [list(map(float, p)) for p in r.polyline], r.edge, r.side,
                       (topology or {}).get(str(r.id)))
            for r in regions if r.geometry == "line" and r.kind == "roof" and r.polyline and len(r.polyline) >= 2]


TOPOLOGY_OFFSET = 0.012     # of the long edge: how far to each side of a line the roof is looked for
_house_roofs: dict = {}     # (photo id, keep flags, planes' mtime) -> the target house's roof, at working size


def _house_roof(store: Store, photo: PhotoRow) -> np.ndarray | None:
    """The union of the target house's roof planes as detection stored them, or None."""
    planes = ((photo.interpreted or {}).get("target") or {}).get("planes") or []
    directory = store.planes_dir(photo.id)
    if not planes or not directory.exists():
        return None
    key = (str(photo.id), tuple(bool(p.get("keep")) for p in planes), directory.stat().st_mtime)
    if key not in _house_roofs:
        masks = store.read_planes(photo.id, len(planes))
        kept = [m for m, p in zip(masks or [], planes) if p.get("keep")]
        if len(_house_roofs) > 8:
            _house_roofs.clear()
        _house_roofs[key] = np.logical_or.reduce(kept) if kept else None
    return _house_roofs[key]


def _topology(store: Store, photo: PhotoRow, regions) -> dict:
    """Per roof line: "shared" when the house's roof lies on both its sides (a hip, a valley, a
    ridge), "free" when on one (an eave, a rake), else unknown. The roof planes' adjacency, which
    tells a hip from a rake better than an edge label does."""
    roof = _house_roof(store, photo)
    if roof is None:
        return {}
    height, width = roof.shape
    fx, fy = width / photo.width, height / photo.height
    offset = max(4.0, TOPOLOGY_OFFSET * max(photo.width, photo.height))
    out = {}
    for region in regions:
        if region.geometry != "line" or region.kind != "roof" or not region.polyline or len(region.polyline) < 2:
            continue
        sides: list[list[bool]] = [[], []]
        for a, b in zip(region.polyline[:-1], region.polyline[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            length = float(np.linalg.norm(b - a))
            if length < 1e-6:
                continue
            normal = np.array([-(b - a)[1], (b - a)[0]]) / length
            for t in np.linspace(0.2, 0.8, 7):
                middle = a + t * (b - a)
                for index, sign in enumerate((1.0, -1.0)):
                    x, y = middle + sign * offset * normal
                    xi, yi = int(x * fx), int(y * fy)
                    sides[index].append(0 <= xi < width and 0 <= yi < height and bool(roof[yi, xi]))
        if not sides[0]:
            continue
        shares = sorted((float(np.mean(sides[0])), float(np.mean(sides[1]))))
        if shares[0] >= 0.6:
            out[str(region.id)] = "shared"
        elif shares[1] >= 0.6 and shares[0] <= 0.25:
            out[str(region.id)] = "free"
    return out


FEATURE_KINDS = ("door", "garage_door")


def _features(store: Store, photo: PhotoRow, regions) -> list[dict]:
    """Detected doors and garage doors as the scale checks read them: the top and bottom of each
    in the middle, and its left and right at mid-height, in photo pixels."""
    out = []
    for region in regions:
        if region.kind not in FEATURE_KINDS or region.geometry != "area" or not region.mask_url:
            continue
        mask = store.read_mask(photo.id, region.id)
        if mask is None or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        top, bottom = int(ys.min()), int(ys.max())
        middle = (top + bottom) / 2
        band = np.abs(ys - middle) <= max(2.0, 0.02 * (bottom - top))
        centre = float(np.median(xs))
        out.append({"id": str(region.id), "kind": region.kind, "label": region.label,
                    "top": [centre, float(top)], "bottom": [centre, float(bottom)],
                    "left": [float(xs[band].min()), float(middle)], "right": [float(xs[band].max()), float(middle)]})
    return out


def _solve(store: Store, photo: PhotoRow, regions, trace: scale.Trace | None = None) -> tuple[camera.Camera, scale.Solution]:
    found = _camera(store, photo)
    solved = scale.solve(found, roof_lines(regions, _topology(store, photo, regions)),
                         (photo.scale or {}).get("measurement"), photo.interpreted, get_settings().roofline_vp_deg,
                         (photo.scale or {}).get("overrides"), trace=trace, features=_features(store, photo, regions),
                         check=(photo.scale or {}).get("check"))
    # The gables' rakes may have rejected the stored camera for another: everything after the
    # solve (overlays, bulbs, wreaths) must use the one the lengths came from.
    return solved.camera or found, solved


@router.get("/{photo_id}/geometry")
def geometry(photo_id: str, store: Store = Depends(get_store)):
    """Each wall's distance and each roof line's real length and 3-D position, with the method
    and any assumption behind it. Computed from the stored camera, measurement and lines on
    every request: an edit changes it at once, and nothing here is billed."""
    photo = store.photo(photo_id)
    regions = store.regions(photo)
    trace = scale.Trace() if get_settings().scale_debug else None
    found, solved = _solve(store, photo, regions, trace)
    measurement = (photo.scale or {}).get("measurement") or {}
    reference = next((r for r in regions if str(r.id) == measurement.get("region_id")), None)
    edited = bool(reference and measurement.get("line_points") and reference.polyline != measurement["line_points"])
    labels = {str(r.id): r.label for r in regions}
    out = solved.to_dict()
    for check in out["checks"]:
        check["labels"] = [labels.get(line_id, line_id) for line_id in check.get("lines", [])]
    if trace is not None:
        directory = store.photo_dir(photo.id) / "debug"
        lines = roof_lines(regions, _topology(store, photo, regions))
        record = scale_debug.record(found, lines, photo.scale, trace, str(photo.id), labels)
        names = scale_debug.write(store.load_image(photo), found, lines, solved, trace, record, directory)
        stamp = record["written_at"].replace(" ", "T")
        out["debug"] = {**record, "files": {name: f"/files/photos/{photo.id}/debug/{name}?at={stamp}" for name in names}}
    return {"reference_edited": edited, **out, "camera": {"focal_source": found.focal_source,
                                                         "focal_35mm": round(found.focal_35mm, 1), "facades": found.facades,
                                                         "focal_checks": found.focal_checks}}



@router.get("/{photo_id}/placement")
def placement_(photo_id: str, store: Store = Depends(get_store)):
    """Every bulb of the confirmed lighting on a line, in photo pixels with its drawn size and
    colour, and each line's count (hidden bulbs included, flagged). Computed on request from the
    camera, the measurement, the lines and the design; nothing here is billed."""
    photo = store.photo(photo_id)
    design = store.design(photo)
    regions = store.regions(photo)
    found, solved = _solve(store, photo, regions)
    rows = {str(row.region_id): row.settings for row in store.design_regions(design)} if design else {}
    catalogue = {str(p.id): p for p in [*store.products(), *store.products(archived=True)]}
    overrides = (photo.scale or {}).get("overrides") or {}
    placed = placement.place(found, regions, rows, catalogue, solved, overrides)
    # Areas need their masks: lit trunks, canopies, fronds and hedges, and every wreath.
    masks = {str(r.id): store.read_mask(photo.id, r.id) for r in regions
             if r.geometry == "area" and (str(r.id) in rows or r.kind == "wreath")}
    placed.update(placement.place_areas(found, regions, rows, catalogue, solved, overrides,
                                        {k: v for k, v in masks.items() if v is not None}))
    return {"regions": placed, "needs": solved.needs or (None if design else "lighting")}
