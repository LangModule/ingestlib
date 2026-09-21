"""collections — the declared taxonomy (rule + attached extract schema)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007_collections"
down_revision = "0006_extractions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("name", sa.String(), primary_key=True),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("schema", postgresql.JSONB()),
        sa.Column("auto_extract", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    op.drop_table("collections")
