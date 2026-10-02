"""products: mini-light density for areas

Revision ID: f7c3a9e2b615
Revises: e5b1c7d9f482
Create Date: 2026-10-01 10:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f7c3a9e2b615"
down_revision: str | Sequence[str] | None = "e5b1c7d9f482"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Lukas (M3): mini lights on a canopy, fronds or a hedge come at light / medium / heavy
    # density; medium is 100 bulbs per 10 sq ft on the built-in minis, light half, heavy double.
    op.add_column("products", sa.Column("density_per_10sqft", sa.Float(), nullable=True))
    op.execute("UPDATE products SET density_per_10sqft = 100.0 WHERE owner_id IS NULL AND category = 'mini'")


def downgrade() -> None:
    op.drop_column("products", "density_per_10sqft")
