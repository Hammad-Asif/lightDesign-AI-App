"""photos: EXIF focal length and the salesperson's scale measurement

Revision ID: d2f6a8c1e347
Revises: a1c7e3f9b240
Create Date: 2026-09-30 10:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d2f6a8c1e347"
down_revision: str | Sequence[str] | None = "a1c7e3f9b240"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # M3 scale and perspective: the lens focal length when the upload carried one (usually it
    # does not), and the one known length that makes the photo's walls metric. Both nullable.
    op.add_column("photos", sa.Column("focal_35mm", sa.Float(), nullable=True))
    op.add_column("photos", sa.Column("scale", postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("photos", "scale")
    op.drop_column("photos", "focal_35mm")
