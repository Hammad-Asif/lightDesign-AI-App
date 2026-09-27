"""products: archive instead of delete

A saved design stores its product's id inside design_regions.settings, a JSONB column no
foreign key can guard. Deleting a product would leave those designs pointing at nothing, so
a product the user is finished with is archived: hidden from the library and from the
instruction parser, still resolvable for the designs that already reference it.

Revision ID: 8c4f2b7e1d05
Revises: 7b3e9d2a5c10
Create Date: 2026-09-27 10:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "8c4f2b7e1d05"
down_revision: str | Sequence[str] | None = "7b3e9d2a5c10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "products",
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    op.drop_column("products", "archived")
