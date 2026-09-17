from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from app.auth import current_user
from app.models import User
from app.schemas import PaletteColour, Product, ProductCategory, ProductCreate, ProductPatch
from app.services.products import COLOURS
from app.storage import Store, get_store, product_dict, product_fields

router = APIRouter(prefix="/products", tags=["products"])


@router.get("/palette", response_model=list[PaletteColour])
def palette(_: User = Depends(current_user)):
    return [{"key": key, "name": name, "hex": hex_} for key, (name, hex_) in COLOURS.items()]


@router.get("", response_model=list[Product])
def list_products(category: ProductCategory | None = None, store: Store = Depends(get_store)):
    return [product_dict(product) for product in store.products(category)]


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
    store.delete_product(store.product(product_id, editable=True))
