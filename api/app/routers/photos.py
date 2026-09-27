import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models import Photo as PhotoRow
from app.schemas import DesignOut, DesignPut, InterpretRequest, ModeRequest, Photo, PhotoDetail
from app.services import lighting, patterns
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
