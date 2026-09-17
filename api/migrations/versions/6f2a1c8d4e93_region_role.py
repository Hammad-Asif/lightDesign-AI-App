"""regions: target or blocker

Revision ID: 6f2a1c8d4e93
Revises: 359c845f6a77
Create Date: 2026-09-17 22:30:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6f2a1c8d4e93"
down_revision: str | Sequence[str] | None = "359c845f6a77"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Palms and trees are detected for any roof request now, so a line passing behind one can be
    # joined and reported. They are stored with the roofline, marked as what they are.
    op.add_column(
        "regions",
        sa.Column("role", sa.String(length=8), server_default="target", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("regions", "role")
