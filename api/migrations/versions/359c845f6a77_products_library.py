"""products: the lighting product library, with a built-in catalogue

Revision ID: 359c845f6a77
Revises: 5d1f0a7c9b21
Create Date: 2026-09-17 20:00:00
"""
import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "359c845f6a77"
down_revision: str | Sequence[str] | None = "5d1f0a7c9b21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Common retail specifications, standing in until the client's own product list arrives. Kept
# here rather than imported from the app, so this migration seeds the same rows forever.
SOCKET_SPACINGS = [12.0, 15.0, 18.0, 24.0, 36.0]
MINI = {"category": "mini", "bulb_diameter_in": 0.2, "spacing_in": 4.0,
        "strand_length_ft": 24.0, "bulb_count": 70}
ICICLE = {"category": "icicle", "bulb_diameter_in": 0.2, "spacing_in": 4.0,
          "strand_length_ft": 7.5, "bulb_count": 70, "drops_in": [6.0, 9.0, 12.0]}
CATALOGUE = [
    {"name": "C9 LED", "category": "c9", "bulb_diameter_in": 1.125, "spacing_in": 12.0,
     "spacing_options_in": SOCKET_SPACINGS},
    {"name": "C9 LED, twinkle", "category": "c9", "bulb_diameter_in": 1.125, "spacing_in": 12.0,
     "spacing_options_in": SOCKET_SPACINGS, "twinkle": True},
    {"name": "C7 LED", "category": "c7", "bulb_diameter_in": 0.875, "spacing_in": 12.0,
     "spacing_options_in": SOCKET_SPACINGS},
    {**MINI, "name": "Mini lights, warm white", "colours": ["warm_white"]},
    {**MINI, "name": "Mini lights, cool white", "colours": ["cool_white"]},
    {**MINI, "name": "Mini lights, red", "colours": ["red"]},
    {**MINI, "name": "Mini lights, green", "colours": ["green"]},
    {**MINI, "name": "Mini lights, multicolour", "colours": ["red", "green", "blue", "amber"]},
    {**MINI, "name": "Mini lights, warm white and cool white", "colours": ["warm_white", "cool_white"]},
    {**MINI, "name": "Mini lights, red and green", "colours": ["red", "green"]},
    {**MINI, "name": "Mini lights, warm white, twinkle", "colours": ["warm_white"], "twinkle": True},
    {**ICICLE, "name": "Icicle lights, warm white", "colours": ["warm_white"]},
    {**ICICLE, "name": "Icicle lights, cool white", "colours": ["cool_white"]},
    {"name": "Wreath, 24 in, warm white", "category": "wreath", "diameter_in": 24.0,
     "bulb_count": 50, "colours": ["warm_white"]},
    {"name": "Wreath, 36 in, warm white", "category": "wreath", "diameter_in": 36.0,
     "bulb_count": 100, "colours": ["warm_white"]},
    {"name": "Wreath, 48 in, warm white", "category": "wreath", "diameter_in": 48.0,
     "bulb_count": 150, "colours": ["warm_white"]},
    {"name": "Garland, 9 ft x 10 in, warm white", "category": "garland", "diameter_in": 10.0,
     "strand_length_ft": 9.0, "bulb_count": 50, "colours": ["warm_white"]},
    {"name": "Garland, 25 ft x 14 in, warm white", "category": "garland", "diameter_in": 14.0,
     "strand_length_ft": 25.0, "bulb_count": 150, "colours": ["warm_white"]},
]


def upgrade() -> None:
    products = op.create_table(
        "products",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=True),
        sa.Column("category", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("bulb_diameter_in", sa.Float(), nullable=True),
        sa.Column("spacing_in", sa.Float(), nullable=True),
        sa.Column("spacing_options_in", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("strand_length_ft", sa.Float(), nullable=True),
        sa.Column("diameter_in", sa.Float(), nullable=True),
        sa.Column("bulb_count", sa.Integer(), nullable=True),
        sa.Column("drops_in", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("colours", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("twinkle", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_products_owner_id"), "products", ["owner_id"], unique=False)
    op.create_index(op.f("ix_products_category"), "products", ["category"], unique=False)

    # Stable ids, so a design that references a built-in product keeps pointing at it.
    rows = [
        {
            "spacing_options_in": [], "drops_in": [], "colours": [], "twinkle": False,
            "bulb_diameter_in": None, "spacing_in": None, "strand_length_ft": None,
            "diameter_in": None, "bulb_count": None,
            **item,
            "id": uuid.uuid5(uuid.NAMESPACE_URL, f"lightdesign/products/{item['name']}"),
            "owner_id": None,
        }
        for item in CATALOGUE
    ]
    op.bulk_insert(products, rows)


def downgrade() -> None:
    op.drop_index(op.f("ix_products_category"), table_name="products")
    op.drop_index(op.f("ix_products_owner_id"), table_name="products")
    op.drop_table("products")
