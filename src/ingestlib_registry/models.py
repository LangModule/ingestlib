"""Registry schema — ingestlib's metadata tables (Postgres).

One row per doc_id in `documents`; pages/regions/sections/chunks/extractions hang off
it (ON DELETE CASCADE). `collections` is the declared taxonomy. Blobs (source bytes,
page renders, figure crops) stay in the blob store, referenced by doc_id.
"""
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_EMPTY = text("''")
_ZERO = text("0")
_FALSE = text("false")
_TEXT = text("'text'")


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    namespace: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    source_path: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    filename: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    source_format: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    source_metadata: Mapped[dict | None] = mapped_column(JSONB)  # source file properties: title, author, subject, dates
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)
    section_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )  # bumped by the ORM on every write — re-ingest, recollect, embed, status change
    status: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    replaced_doc_id: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)

    category: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    classify_confidence: Mapped[float | None] = mapped_column(Float)
    classify_reasoning: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)
    classify_alternatives: Mapped[list | None] = mapped_column(JSONB)  # [{label, score}]
    collection: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)

    vector_store: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    vector_dim: Mapped[int | None] = mapped_column(Integer)
    vector_count: Mapped[int | None] = mapped_column(Integer)
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_documents_ns_path", "namespace", "source_path"),
        Index("ix_documents_collection", "collection"),
        Index("ix_documents_category", "category"),
        Index("ix_documents_namespace", "namespace"),
    )


class Page(Base):
    __tablename__ = "pages"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    page_num: Mapped[int] = mapped_column(Integer, primary_key=True)
    width: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)
    height: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)


class Region(Base):
    __tablename__ = "regions"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    page_num: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_type: Mapped[str] = mapped_column(String, nullable=False, server_default=_TEXT)
    bbox: Mapped[dict] = mapped_column(JSONB, nullable=False)  # {x, y, width, height}
    text: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)

    __table_args__ = (Index("ix_regions_type", "doc_id", "region_type"),)


class Section(Base):
    __tablename__ = "sections"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    ord: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)
    pages: Mapped[list | None] = mapped_column(JSONB)


class Chunk(Base):
    __tablename__ = "chunks"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    section: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    heading: Mapped[str] = mapped_column(String, nullable=False, server_default=_EMPTY)
    kind: Mapped[str] = mapped_column(String, nullable=False, server_default=_TEXT)
    markdown: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)
    embedding_text: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)
    pages: Mapped[list | None] = mapped_column(JSONB)
    region_ids: Mapped[dict | None] = mapped_column(JSONB)  # {page_num: [region_id]}
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)

    __table_args__ = (Index("ix_chunks_section", "doc_id", "section"),)


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"))
    schema_name: Mapped[str] = mapped_column(String, nullable=False)
    item_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=_ZERO)
    value: Mapped[dict | None] = mapped_column(JSONB)
    fields: Mapped[dict | None] = mapped_column(JSONB)  # {field: {confidence, region_ids, pages, grounded}}
    pages: Mapped[list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_extractions_doc_schema", "doc_id", "schema_name"),)


class Collection(Base):
    __tablename__ = "collections"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=_EMPTY)
    extract_schema: Mapped[dict | None] = mapped_column("schema", JSONB)
    auto_extract: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=_FALSE)
