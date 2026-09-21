"""sections — split sections (name + description + pages)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_sections"
down_revision = "0003_regions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sections",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("pages", postgresql.JSONB()),
        sa.PrimaryKeyConstraint("doc_id", "ord"),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("sections")
