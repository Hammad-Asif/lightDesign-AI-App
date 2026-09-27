"""products: a picture of the product

What a C9 LED or a 24 in wreath looks like, so the salesperson picks by sight rather than by
reading specifications. The file lives beside the photos under storage_dir/products; the
column holds the URL the browser asks for, exactly as regions.mask_url does for masks.

Revision ID: 9d5a3c8f2e16
Revises: 8c4f2b7e1d05
Create Date: 2026-09-27 12:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9d5a3c8f2e16"
down_revision: str | Sequence[str] | None = "8c4f2b7e1d05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("products", sa.Column("image_url", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("products", "image_url")
