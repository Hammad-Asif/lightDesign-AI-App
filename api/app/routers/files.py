from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.storage import Store, get_store

router = APIRouter(prefix="/files", tags=["files"])


# Replaces the old unauthenticated static mount: every photo and mask is served only to the
# photo's owner. Mask URLs carry a version query string, so a short private cache is safe.
@router.get("/photos/{photo_id}/{path:path}")
def photo_file(photo_id: str, path: str, store: Store = Depends(get_store)):
    target = store.file_path(store.photo(photo_id), path)
    return FileResponse(target, headers={"Cache-Control": "private, max-age=3600"})
