"""regions: roof-line confidence and sources

Revision ID: a1c7e3f9b240
Revises: 9d5a3c8f2e16
Create Date: 2026-09-28 12:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1c7e3f9b240"
down_revision: str | Sequence[str] | None = "9d5a3c8f2e16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Detected roof lines: evidence-based confidence in [0, 1] and the sources that found the
    # line (["v3", "partition"]). Both nullable: areas, drawn lines and older rows have none.
    op.add_column("regions", sa.Column("confidence", sa.Float(), nullable=True))
    op.add_column("regions", sa.Column("sources", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("regions", "sources")
    op.drop_column("regions", "confidence")
