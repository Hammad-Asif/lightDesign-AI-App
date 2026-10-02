"""built-in catalogue: Lukas's icicle and garland rules

Revision ID: e5b1c7d9f482
Revises: d2f6a8c1e347
Create Date: 2026-09-30 16:00:00
"""
from collections.abc import Sequence

from alembic import op

revision: str = "e5b1c7d9f482"
down_revision: str | Sequence[str] | None = "d2f6a8c1e347"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Lukas (M3): icicle drops every 6 in, lengths alternating 12 / 18 / 24 in; garland lights
    # every 4 in within its band. Built-in rows only; products a user added are theirs.
    op.execute("UPDATE products SET spacing_in = 6.0, drops_in = '[12.0, 18.0, 24.0]'::jsonb "
               "WHERE owner_id IS NULL AND category = 'icicle'")
    op.execute("UPDATE products SET spacing_in = 4.0 WHERE owner_id IS NULL AND category = 'garland'")


def downgrade() -> None:
    op.execute("UPDATE products SET spacing_in = 4.0, drops_in = '[6.0, 9.0, 12.0]'::jsonb "
               "WHERE owner_id IS NULL AND category = 'icicle'")
    op.execute("UPDATE products SET spacing_in = NULL WHERE owner_id IS NULL AND category = 'garland'")
