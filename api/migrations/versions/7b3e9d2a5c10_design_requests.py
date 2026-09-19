"""designs: the confirmed specification

Revision ID: 7b3e9d2a5c10
Revises: 6f2a1c8d4e93
Create Date: 2026-09-19 10:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7b3e9d2a5c10"
down_revision: str | Sequence[str] | None = "6f2a1c8d4e93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The instruction's requests as the user confirmed them — places, product, pattern, spacing —
    # each with the regions it lit. The per-region settings themselves are in design_regions.
    op.add_column(
        "designs",
        sa.Column("requests", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("designs", "requests")
