"""chunks — retrieval units ({doc_id}:{chunk_id} = the vector id)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_chunks"
down_revision = "0004_sections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("chunk_id", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("heading", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("kind", sa.String(), nullable=False, server_default=sa.text("'text'")),
        sa.Column("markdown", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("embedding_text", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("pages", postgresql.JSONB()),
        sa.Column("region_ids", postgresql.JSONB()),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("doc_id", "chunk_id"),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_chunks_section", "chunks", ["doc_id", "section"])


def downgrade() -> None:
    op.drop_table("chunks")
