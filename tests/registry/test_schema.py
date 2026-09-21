"""Registry schema — structure guards. No server needed; runs in the fast suite.

Locks the table shape (columns, PKs, cascading FKs, indexes, defaults) so a change
to models.py that would break the migration or a bare insert fails a test first.
"""
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from ingestlib_registry.models import Base

_TABLES = {"documents", "pages", "regions", "sections", "chunks", "extractions", "collections"}

# NOT NULL columns the writer always supplies, so a default is neither expected nor wanted.
_NO_DEFAULT_OK = {("regions", "bbox"), ("extractions", "schema_name")}


def _table(name):
    return Base.metadata.tables[name]


def test_all_seven_tables_registered():
    assert set(Base.metadata.tables) == _TABLES


def test_document_core_columns_present():
    cols = set(_table("documents").columns.keys())
    for c in (
        "doc_id", "namespace", "source_path", "filename",
        "created_at", "updated_at",                          # audit timestamps
        "category", "classify_confidence", "classify_reasoning",
        "classify_alternatives", "collection",
        "vector_store", "vector_dim", "vector_count", "embedded_at",
        "page_count", "section_count", "chunk_count",       # the denormalized counts
    ):
        assert c in cols, c


def test_primary_keys():
    def pk(n):
        return tuple(c.name for c in _table(n).primary_key.columns)

    assert pk("documents") == ("doc_id",)
    assert pk("pages") == ("doc_id", "page_num")
    assert pk("regions") == ("doc_id", "page_num", "region_id")
    assert pk("sections") == ("doc_id", "ord")
    assert pk("chunks") == ("doc_id", "chunk_id")
    assert pk("extractions") == ("id",)
    assert pk("collections") == ("name",)


def test_every_foreign_key_cascades_to_documents():
    for name in _TABLES:
        for fk in _table(name).foreign_keys:
            assert fk.column.table.name == "documents"
            assert fk.ondelete == "CASCADE", f"{name}.{fk.parent.name}"


def test_chunks_has_no_text_column():
    # locked decision: chunks.text dropped (derive from markdown)
    assert "text" not in _table("chunks").columns


def test_regions_bbox_is_jsonb_not_null():
    bbox = _table("regions").columns["bbox"]
    assert bbox.nullable is False
    assert "JSONB" in str(bbox.type.compile(dialect=postgresql.dialect()))


def test_expected_indexes_present():
    names = {ix.name for t in Base.metadata.tables.values() for ix in t.indexes}
    for want in (
        "ix_documents_ns_path", "ix_documents_collection", "ix_documents_category",
        "ix_documents_namespace", "ix_regions_type", "ix_chunks_section",
        "ix_extractions_doc_schema",
    ):
        assert want in names, want


def test_extractions_id_is_bigserial():
    ddl = str(CreateTable(_table("extractions")).compile(dialect=postgresql.dialect()))
    assert "BIGSERIAL" in ddl


def test_notnull_columns_have_a_default():
    """Every NOT NULL column that isn't a key or writer-required carries a server_default,
    so a bare INSERT (a migration, a raw write) can't trip the constraint."""
    for name in _TABLES:
        t = _table(name)
        pk_cols = set(t.primary_key.columns.keys())
        fk_cols = {fk.parent.name for fk in t.foreign_keys}
        for col in t.columns:
            if col.nullable or col.name in pk_cols or col.name in fk_cols:
                continue
            if (name, col.name) in _NO_DEFAULT_OK:
                continue
            assert col.server_default is not None, f"{name}.{col.name} is NOT NULL without a default"


def test_ddl_compiles_for_postgres():
    d = postgresql.dialect()
    for t in Base.metadata.tables.values():
        assert str(CreateTable(t).compile(dialect=d)).strip()
