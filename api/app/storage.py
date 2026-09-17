import io
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import Depends, HTTPException
from PIL import Image, ImageOps
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.auth import current_user
from app.config import get_settings
from app.db import get_db
from app.models import DesignRegion, Photo, Product, Region, User
from app.services import occlusion

REGION_FIELDS = (
    "kind", "role", "label", "geometry", "source", "color", "score",
    "visible", "side", "edge", "polyline", "mask_url", "hidden",
)


def _uuid(value: str, missing: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(404, missing)


def photo_dict(photo: Photo) -> dict:
    return {
        "id": str(photo.id),
        "prompt": photo.prompt,
        "width": photo.width,
        "height": photo.height,
        "image_url": f"/files/photos/{photo.id}/original.png",
        "mode": photo.mode,
        "approved": photo.approved,
        "interpreted": photo.interpreted,
    }


def region_dict(region: Region) -> dict:
    return {"id": str(region.id), **{name: getattr(region, name) for name in REGION_FIELDS}}


PRODUCT_FIELDS = (
    "category", "name", "bulb_diameter_in", "spacing_in", "spacing_options_in", "strand_length_ft",
    "diameter_in", "bulb_count", "drops_in", "colours", "twinkle",
)


def product_fields(product: Product) -> dict:
    return {name: getattr(product, name) for name in PRODUCT_FIELDS}


def product_dict(product: Product) -> dict:
    return {"id": str(product.id), "builtin": product.owner_id is None, **product_fields(product)}


class Store:
    """Photo and region records in Postgres, image and mask files on disk, for one user.

    `photo()` is the single place that decides who may see a photo; every route goes through
    it. Phase 2 multi-tenancy widens that one query to an organisation.
    """

    def __init__(self, db: Session, user: User, root: Path):
        self.db = db
        self.user = user
        self.photos = root / "photos"
        self.photos.mkdir(parents=True, exist_ok=True)

    # -- access ----------------------------------------------------------------

    def photo(self, photo_id: str) -> Photo:
        photo = self.db.scalar(
            select(Photo).where(
                Photo.id == _uuid(photo_id, "Photo not found"),
                Photo.owner_id == self.user.id,
            )
        )
        if photo is None:
            raise HTTPException(404, "Photo not found")
        return photo

    def list_photos(self) -> list[Photo]:
        query = select(Photo).where(Photo.owner_id == self.user.id).order_by(Photo.created_at.desc())
        return list(self.db.scalars(query))

    def save(self) -> None:
        self.db.commit()

    def release(self) -> None:
        """End the open read transaction so no connection idles through a slow model call."""
        self.db.commit()

    # -- photos ----------------------------------------------------------------

    def create_photo(self, data: bytes, prompt: str, max_edge: int) -> Photo:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.LANCZOS)

        photo = Photo(
            id=uuid.uuid4(),
            owner_id=self.user.id,
            prompt=prompt,
            width=image.width,
            height=image.height,
        )
        directory = self.photo_dir(photo.id)
        (directory / "masks").mkdir(parents=True, exist_ok=True)
        image.save(directory / "original.png", optimize=True)

        self.db.add(photo)
        self.db.commit()
        return photo

    def update_photo(self, photo: Photo, **fields) -> Photo:
        for name, value in fields.items():
            setattr(photo, name, value)
        self.db.commit()
        return photo

    def load_image(self, photo: Photo) -> Image.Image:
        return Image.open(self.photo_dir(photo.id) / "original.png").convert("RGB")

    # -- regions ---------------------------------------------------------------

    def regions(self, photo: Photo) -> list[Region]:
        query = select(Region).where(Region.photo_id == photo.id).order_by(Region.seq)
        return list(self.db.scalars(query))

    def region(self, photo: Photo, region_id: str, *, lock: bool = False) -> Region:
        query = select(Region).where(
            Region.id == _uuid(region_id, "Region not found"),
            Region.photo_id == photo.id,
        )
        if lock:
            query = query.with_for_update()
        region = self.db.scalar(query)
        if region is None:
            raise HTTPException(404, "Region not found")
        return region

    def add_region(self, photo: Photo, region: Region) -> Region:
        region.photo_id = photo.id
        self.db.add(region)
        self.db.commit()
        return region

    def replace_auto_regions(self, photo: Photo, regions: list[Region]) -> list[Region]:
        """Swap every detected region for a fresh set in one transaction; manual ones stay."""
        self.db.execute(select(Photo.id).where(Photo.id == photo.id).with_for_update())
        stale = list(
            self.db.scalars(
                select(Region.id).where(Region.photo_id == photo.id, Region.source == "auto")
            )
        )
        self.db.execute(delete(Region).where(Region.id.in_(stale)))
        for region in regions:
            region.photo_id = photo.id
        self.db.add_all(regions)
        self.db.commit()
        for region_id in stale:
            self.delete_mask(photo.id, region_id)
        return self.regions(photo)

    def delete_region(self, photo: Photo, region_id: str) -> Region:
        region = self.region(photo, region_id)
        self.db.delete(region)
        self.db.commit()
        self.delete_mask(photo.id, region.id)
        return region

    def refresh_hidden(self, photo: Photo) -> list[Region]:
        """Recompute which stretches of every line pass behind an object.

        The single place hidden stretches come from: called after detection and after any
        change to a line's path or to a blocking object, so correcting an object in the
        editor corrects every line behind it.
        """
        regions = self.regions(photo)
        lines = [r for r in regions if r.geometry == "line" and r.polyline]
        if not lines:
            return regions
        masks = [
            mask
            for r in regions
            if r.geometry == "area" and r.kind in occlusion.BLOCKER_KINDS
            for mask in [self.read_mask(photo.id, r.id)]
            if mask is not None
        ]
        blockers = occlusion.blocker_union(masks, (photo.height, photo.width))
        for line in lines:
            line.hidden = occlusion.hidden_stretches(line.polyline, blockers, photo.width) or None
        self.db.commit()
        return regions

    # -- products --------------------------------------------------------------

    def _products(self):
        """The one query deciding which products a user sees: the built-in catalogue and their own."""
        return select(Product).where(or_(Product.owner_id.is_(None), Product.owner_id == self.user.id))

    def products(self, category: str | None = None) -> list[Product]:
        query = self._products()
        if category:
            query = query.where(Product.category == category)
        return list(self.db.scalars(query.order_by(Product.owner_id.is_not(None), Product.category, Product.name)))

    def product(self, product_id: str, *, editable: bool = False) -> Product:
        product = self.db.scalar(
            self._products().where(Product.id == _uuid(product_id, "Product not found"))
        )
        if product is None:
            raise HTTPException(404, "Product not found")
        if editable and product.owner_id is None:
            raise HTTPException(403, "Built-in products can't be changed. Add your own product instead.")
        return product

    def add_product(self, fields: dict) -> Product:
        product = Product(id=uuid.uuid4(), owner_id=self.user.id, **fields)
        self.db.add(product)
        self.db.commit()
        return product

    def update_product(self, product: Product, fields: dict) -> Product:
        for name, value in fields.items():
            setattr(product, name, value)
        self.db.commit()
        return product

    def delete_product(self, product: Product) -> None:
        # Settings reference products by id inside JSON, where no foreign key can guard them.
        used = self.db.scalar(
            select(DesignRegion.id)
            .where(DesignRegion.settings["product_id"].astext == str(product.id))
            .limit(1)
        )
        if used is not None:
            raise HTTPException(409, "This product is used in a design")
        self.db.delete(product)
        self.db.commit()

    # -- files -----------------------------------------------------------------

    def photo_dir(self, photo_id) -> Path:
        return self.photos / str(photo_id)

    def mask_path(self, photo_id, region_id) -> Path:
        return self.photo_dir(photo_id) / "masks" / f"{region_id}.png"

    def write_mask(self, photo_id, region_id, mask: np.ndarray) -> str:
        path = self.mask_path(photo_id, region_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = (mask.astype(bool) * 255).astype(np.uint8)
        Image.fromarray(array, mode="L").save(path)
        return f"/files/photos/{photo_id}/masks/{region_id}.png?v={time.time_ns()}"

    def write_mask_bytes(self, photo_id, region_id, data: bytes) -> str:
        uploaded = Image.open(io.BytesIO(data))
        alpha = uploaded.getchannel("A") if uploaded.mode == "RGBA" else uploaded.convert("L")
        return self.write_mask(photo_id, region_id, np.array(alpha) > 8)

    def read_mask(self, photo_id, region_id) -> np.ndarray | None:
        path = self.mask_path(photo_id, region_id)
        if not path.exists():
            return None
        return np.array(Image.open(path).convert("L")) > 127

    def delete_mask(self, photo_id, region_id) -> None:
        self.mask_path(photo_id, region_id).unlink(missing_ok=True)

    def file_path(self, photo: Photo, relative: str) -> Path:
        """Resolve a file inside one photo's directory, refusing anything that escapes it."""
        base = self.photo_dir(photo.id).resolve()
        target = (base / relative).resolve()
        if base not in target.parents or not target.is_file():
            raise HTTPException(404, "File not found")
        return target


def get_store(db: Session = Depends(get_db), user: User = Depends(current_user)) -> Store:
    return Store(db, user, get_settings().storage_dir)
