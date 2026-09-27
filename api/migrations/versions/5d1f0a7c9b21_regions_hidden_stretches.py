"""regions: hidden stretches behind blocking objects

Revision ID: 5d1f0a7c9b21
Revises: c483e50f6062
Create Date: 2026-09-15 10:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "5d1f0a7c9b21"
down_revision: str | Sequence[str] | None = "c483e50f6062"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # [start, end] pixel distances along a line where it passes behind an object. Derived
    # from the blocking objects' masks and recomputed whenever a line or an object changes.
    op.add_column(
        "regions",
        sa.Column("hidden", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("regions", "hidden")
