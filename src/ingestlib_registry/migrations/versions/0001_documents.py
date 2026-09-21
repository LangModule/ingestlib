"""documents — the core row (one per doc_id)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_documents"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(), primary_key=True),
        sa.Column("namespace", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("source_path", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("filename", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("source_format", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("source_metadata", postgresql.JSONB()),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("section_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("status", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("replaced_doc_id", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("category", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("classify_confidence", sa.Float()),
        sa.Column("classify_reasoning", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("classify_alternatives", postgresql.JSONB()),
        sa.Column("collection", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("vector_store", sa.String(), nullable=False, server_default=sa.text("''")),
        sa.Column("vector_dim", sa.Integer()),
        sa.Column("vector_count", sa.Integer()),
        sa.Column("embedded_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_documents_ns_path", "documents", ["namespace", "source_path"])
    op.create_index("ix_documents_collection", "documents", ["collection"])
    op.create_index("ix_documents_category", "documents", ["category"])
    op.create_index("ix_documents_namespace", "documents", ["namespace"])


def downgrade() -> None:
    op.drop_table("documents")
