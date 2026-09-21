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
from app.models import Design, DesignRegion, Photo, Product, Region, User
from app.services import occlusion, partition

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
                select(Region.id).where(Region.photo_id == photo.id, Region.source.in_(("auto", "inferred")))
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
        blockers = self._blocker_union(photo, regions)
        for line in lines:
            line.hidden = occlusion.hidden_stretches(line.polyline, blockers, photo.width) or None
        self.db.commit()
        return regions

    def _blocker_union(self, photo: Photo, regions: list[Region]) -> np.ndarray | None:
        masks = [
            mask
            for r in regions
            if r.geometry == "area" and r.kind in occlusion.BLOCKER_KINDS
            for mask in [self.read_mask(photo.id, r.id)]
            if mask is not None
        ]
        return occlusion.blocker_union(masks, (photo.height, photo.width))

    def rejoin_behind_blockers(self, photo: Photo) -> list[Region]:
        """After an object in front of the house is painted in or corrected: mend the roof lines
        behind it the way detection would have.

        A detected roof line that the object now cuts apart is joined back into one straight
        line, and an end the object cuts off is carried through to its far side — the same two
        steps roof_lines applies at detection, on the stored lines. So when the model misses a
        palm, painting the trunk is enough: the roofline behind it mends itself. Only detected
        lines take part (auto and inferred); a line someone drew is never touched, and a line
        never un-joins when an object is erased — that is a deliberate correction, kept.
        """
        regions = self.regions(photo)
        lines = [
            r for r in regions
            if r.geometry == "line" and r.kind == "roof" and r.source in ("auto", "inferred")
            and r.polyline and len(r.polyline) >= 2
        ]
        blockers = self._blocker_union(photo, regions)
        if blockers is None or not lines:
            return self.refresh_hidden(photo)

        points = np.array([p for r in lines for p in r.polyline], dtype=float)
        extent = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0))
        absorbed: set = set()
        while True:
            best = None
            for left in lines:
                if left.id in absorbed:
                    continue
                a = np.asarray(left.polyline)
                if not occlusion._rightward(a):
                    continue
                for right in lines:
                    if right is left or right.id in absorbed or right.edge != left.edge:
                        continue
                    b = np.asarray(right.polyline)
                    if not occlusion._rightward(b):
                        continue
                    gap = occlusion._gap_if_continues(a, b, blockers, photo.width, partition.BRIDGE_OFFSET_L * extent)
                    if gap is not None and (best is None or gap < best[0]):
                        best = (gap, left, right)
            if best is None:
                break
            _, left, right = best
            # Detected lines are straight fits, so the join is the two outer ends; a wobblier
            # polyline keeps its points, as occlusion.bridge would.
            if len(left.polyline) == 2 and len(right.polyline) == 2:
                left.polyline = [list(map(int, left.polyline[0])), list(map(int, right.polyline[-1]))]
            else:
                left.polyline = [list(map(int, p)) for p in left.polyline] + [list(map(int, p)) for p in right.polyline]
            absorbed.add(right.id)

        # Nothing is carried past an object here: no roof mask is stored, and a line is only
        # carried to where the masks say there is roof. Joining the pieces either side of the
        # object is what a painted object can do from geometry alone.
        for line in lines:
            if line.id in absorbed:
                self.db.delete(line)
        self.db.commit()
        return self.refresh_hidden(photo)

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

    # -- designs ---------------------------------------------------------------

    def design(self, photo: Photo) -> Design | None:
        """The photo's design: one per photo until Milestone 4 adds revisions."""
        return self.db.scalar(
            select(Design)
            .where(Design.photo_id == photo.id, Design.owner_id == self.user.id)
            .order_by(Design.created_at.desc())
            .limit(1)
        )

    def design_regions(self, design: Design) -> list[DesignRegion]:
        return list(self.db.scalars(select(DesignRegion).where(DesignRegion.design_id == design.id)))

    def set_design(self, photo: Photo, requests: list[dict], rows: dict[uuid.UUID, dict]) -> Design:
        """Replace the photo's design with the confirmed requests and their per-region settings."""
        design = self.design(photo)
        if design is None:
            design = Design(id=uuid.uuid4(), owner_id=self.user.id, photo_id=photo.id)
            self.db.add(design)
        else:
            self.db.execute(delete(DesignRegion).where(DesignRegion.design_id == design.id))
        design.requests = requests
        self.db.add_all(
            DesignRegion(id=uuid.uuid4(), design_id=design.id, region_id=region_id, settings=settings)
            for region_id, settings in rows.items()
        )
        self.db.commit()
        return design

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
