"""regions — parse structure (one row per region, queryable provenance)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_regions"
down_revision = "0002_pages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regions",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("page_num", sa.Integer(), nullable=False),
        sa.Column("region_id", sa.Integer(), nullable=False),
        sa.Column("region_type", sa.String(), nullable=False, server_default=sa.text("'text'")),
        sa.Column("bbox", postgresql.JSONB(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("content", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.PrimaryKeyConstraint("doc_id", "page_num", "region_id"),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_regions_type", "regions", ["doc_id", "region_type"])


def downgrade() -> None:
    op.drop_table("regions")
