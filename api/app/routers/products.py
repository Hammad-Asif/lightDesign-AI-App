from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import ValidationError

from app.auth import current_user
from app.config import get_settings
from app.models import User
from app.schemas import PaletteColour, Product, ProductCategory, ProductCreate, ProductPatch
from app.services.products import COLOURS
from app.storage import Store, get_store, product_dict, product_fields

router = APIRouter(prefix="/products", tags=["products"])

# The same formats a customer photo may be uploaded in.
ALLOWED_IMAGES = {"image/jpeg", "image/png", "image/webp"}


@router.get("/palette", response_model=list[PaletteColour])
def palette(_: User = Depends(current_user)):
    return [{"key": key, "name": name, "hex": hex_} for key, (name, hex_) in COLOURS.items()]


@router.get("", response_model=list[Product])
def list_products(category: ProductCategory | None = None, archived: bool = False,
                  store: Store = Depends(get_store)):
    """The library. `archived=true` lists the retired products instead, so they can be brought back."""
    return [product_dict(product) for product in store.products(category, archived=archived)]


@router.post("", response_model=Product, status_code=201)
def create_product(body: ProductCreate, store: Store = Depends(get_store)):
    return product_dict(store.add_product(body.model_dump()))


@router.get("/{product_id}", response_model=Product)
def get_product(product_id: str, store: Store = Depends(get_store)):
    return product_dict(store.product(product_id))


@router.patch("/{product_id}", response_model=Product)
def update_product(product_id: str, body: ProductPatch, store: Store = Depends(get_store)):
    product = store.product(product_id, editable=True)
    # A change is checked as the whole product it produces: new colours on a C9 line, or a
    # mini strand losing its spacing, is refused even when each field looks fine on its own.
    merged = {**product_fields(product), **body.model_dump(exclude_unset=True)}
    try:
        valid = ProductCreate.model_validate(merged)
    except ValidationError as exc:
        raise HTTPException(422, [error["msg"] for error in exc.errors()])
    return product_dict(store.update_product(product, valid.model_dump()))


@router.delete("/{product_id}", status_code=204)
def delete_product(product_id: str, store: Store = Depends(get_store)):
    """Remove the product, or archive it when a saved design already names it."""
    store.delete_product(store.product(product_id, editable=True))


@router.post("/{product_id}/restore", response_model=Product)
def restore_product(product_id: str, store: Store = Depends(get_store)):
    return product_dict(store.restore_product(store.product(product_id, editable=True)))


@router.put("/{product_id}/image", response_model=Product)
def put_product_image(product_id: str, file: UploadFile = File(...),
                      store: Store = Depends(get_store)):
    """A picture of the product. Built-in entries are read-only, as they are for every field."""
    settings = get_settings()
    if file.content_type not in ALLOWED_IMAGES:
        raise HTTPException(415, "Upload a JPG, PNG or WEBP image")
    data = file.file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image exceeds {settings.max_upload_mb} MB")
    product = store.product(product_id, editable=True)
    try:
        return product_dict(store.write_product_image(product, data))
    except OSError:
        raise HTTPException(400, "Could not read that image file")


@router.delete("/{product_id}/image", response_model=Product)
def remove_product_image(product_id: str, store: Store = Depends(get_store)):
    return product_dict(store.delete_product_image(store.product(product_id, editable=True)))
