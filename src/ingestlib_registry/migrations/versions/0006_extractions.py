"""extractions — extract output, persisted on demand."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006_extractions"
down_revision = "0005_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extractions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("schema_name", sa.String(), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("value", postgresql.JSONB()),
        sa.Column("fields", postgresql.JSONB()),
        sa.Column("pages", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_extractions_doc_schema", "extractions", ["doc_id", "schema_name"])


def downgrade() -> None:
    op.drop_table("extractions")
