"""Database tables.

Images and masks stay on disk under STORAGE_DIR; these rows are the records that point at
them. Every top-level resource is owned by a user, and all access is scoped through
`storage.Store` — Phase 2 multi-tenancy adds an organisation to that one scope rather than
rewriting access rules across the codebase.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Timestamped:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(Timestamped, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Stored lower-cased, so the unique index is effectively case-insensitive.
    email: Mapped[str] = mapped_column(String(320), unique=True)
    name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=text("true"))


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    # Only a SHA-256 of the cookie token is kept, so a leaked table cannot be replayed.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    user: Mapped[User] = relationship(lazy="joined")


class Photo(Timestamped, Base):
    __tablename__ = "photos"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    prompt: Mapped[str] = mapped_column(Text, default="", server_default="")
    width: Mapped[int]
    height: Mapped[int]
    mode: Mapped[str | None] = mapped_column(String(16))
    approved: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    interpreted: Mapped[dict | None] = mapped_column(JSONB)


class Region(Timestamped, Base):
    __tablename__ = "regions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("photos.id", ondelete="CASCADE"), index=True
    )
    # Creation order — the editor lists regions in the order they were made.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(), unique=True)
    kind: Mapped[str] = mapped_column(String(32))
    # target = something to light. blocker = something standing in front of the house, detected
    # so a line behind it can be joined and its hidden stretch reported; never lit itself.
    role: Mapped[str] = mapped_column(String(8), default="target", server_default="target")
    label: Mapped[str] = mapped_column(String(200))
    geometry: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(8))
    color: Mapped[str] = mapped_column(String(7))
    score: Mapped[float] = mapped_column(Float, default=1.0)
    visible: Mapped[bool] = mapped_column(default=True, server_default=text("true"))
    side: Mapped[str | None] = mapped_column(String(8))
    edge: Mapped[str | None] = mapped_column(String(8))
    polyline: Mapped[list | None] = mapped_column(JSONB)
    mask_url: Mapped[str | None] = mapped_column(Text)
    # Lines only: [start, end] pixel distances along the polyline that pass behind an object.
    # Derived from the blocking objects' masks by Store.refresh_hidden, never set directly.
    hidden: Mapped[list | None] = mapped_column(JSONB)


class Product(Timestamped, Base):
    """A lighting product. See services/products.py for what each category must describe."""

    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # None for the built-in catalogue, which every user reads and nobody edits; otherwise the
    # user who added it, and only they see it.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    category: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(120))
    bulb_diameter_in: Mapped[float | None] = mapped_column(Float)
    # Centre to centre along the wire; for icicles, between drops.
    spacing_in: Mapped[float | None] = mapped_column(Float)
    # Socket lines can be ordered at several spacings; empty means spacing_in is fixed.
    spacing_options_in: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    strand_length_ft: Mapped[float | None] = mapped_column(Float)
    # Wreath diameter, or a garland's thickness.
    diameter_in: Mapped[float | None] = mapped_column(Float)
    bulb_count: Mapped[int | None]
    drops_in: Mapped[list] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    # The colour sequence built into the product, repeating along it. Empty for socket lines,
    # whose bulbs are chosen per design.
    colours: Mapped[list] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    twinkle: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    # Retired rather than deleted: design_regions.settings holds this id inside JSONB, where no
    # foreign key can protect it. An archived product leaves the library and the parser, and
    # still resolves for the designs that already chose it.
    archived: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    # A picture of the product, so it can be chosen by sight. The file sits under
    # storage_dir/products; this is the URL for it, carrying a version so a replacement shows
    # immediately. None when none was uploaded.
    image_url: Mapped[str | None] = mapped_column(Text)


class Design(Timestamped, Base):
    __tablename__ = "designs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("photos.id", ondelete="CASCADE"), index=True
    )
    # A revision (Milestone 4) is a new design pointing at the one it revises.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("designs.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(200), default="Design", server_default="Design")
    # The confirmed specification: the instruction's requests as the user accepted or corrected
    # them, each with the region ids it lit. design_regions holds the per-region settings.
    requests: Mapped[list | None] = mapped_column(JSONB)


class DesignRegion(Timestamped, Base):
    __tablename__ = "design_regions"
    __table_args__ = (UniqueConstraint("design_id", "region_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    design_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("designs.id", ondelete="CASCADE"), index=True
    )
    region_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("regions.id", ondelete="CASCADE"), index=True
    )
    # Product, colour, spacing and pattern for this region. They are render parameters,
    # never baked into pixels; the product library (Day 4) defines their shape.
    settings: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
