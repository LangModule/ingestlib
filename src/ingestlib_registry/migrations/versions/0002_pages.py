"""pages — per-page dimensions (bbox coordinate system)."""
import sqlalchemy as sa
from alembic import op

revision = "0002_pages"
down_revision = "0001_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pages",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("page_num", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("height", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("doc_id", "page_num"),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("pages")
