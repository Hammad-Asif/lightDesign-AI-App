"""photos: a project name (the customer or the address) for the projects list

Revision ID: a8d4e2c7f913
Revises: f7c3a9e2b615
Create Date: 2026-10-07 12:00:00
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8d4e2c7f913"
down_revision: str | Sequence[str] | None = "f7c3a9e2b615"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Empty for photos uploaded before: the projects list falls back to their instruction.
    op.add_column("photos", sa.Column("name", sa.String(200), nullable=False, server_default=""))


def downgrade() -> None:
    op.drop_column("photos", "name")
