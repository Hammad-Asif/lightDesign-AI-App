from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models import Photo as PhotoRow
from app.schemas import ModeRequest, Photo, PhotoDetail
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
        return photo_dict(store.create_photo(data, prompt.strip(), settings.max_image_edge))
    except OSError:
        raise HTTPException(400, "Could not read that image file")


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
