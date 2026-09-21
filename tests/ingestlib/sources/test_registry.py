"""resolve_sources() → Source instances from sources.yaml, cached by name.

Building a source never connects (only answer()/health() do), so these run
ungated against a scratch config — no database is touched."""
import pytest

from ingestlib.sources.documents import DocumentSource
from ingestlib.sources.sql.source import SqlSource

# ollama/local/sqlite/none needs no AWS section (see test_config's zero-cloud case)
_CONFIG = """\
llm_provider: ollama
embedding_provider: ollama
artifact_store: local
vector_store: sqlite
reranker: none
"""

_SOURCES = """\
warehouse:
  type: sqlite
  dsn: sqlite:///{db}
  description: the analytics warehouse
  tables:
    rx: prescriptions
corpus:
  type: documents
  namespace: tenant-a
"""


def _setup(scratch_config, db_path) -> None:
    (scratch_config / "config.yaml").write_text(_CONFIG)
    (scratch_config / "sources.yaml").write_text(_SOURCES.format(db=db_path))


def test_resolve_builds_sql_and_document_sources(scratch_config, tmp_path):
    _setup(scratch_config, tmp_path / "w.db")
    from ingestlib.sources.registry import resolve_sources

    sql, doc = resolve_sources(["warehouse", "corpus"])
    assert isinstance(sql, SqlSource) and sql.name == "warehouse"
    assert isinstance(doc, DocumentSource) and doc.name == "corpus"


def test_instances_are_cached(scratch_config, tmp_path):
    _setup(scratch_config, tmp_path / "w.db")
    from ingestlib.sources.registry import resolve_sources

    assert resolve_sources(["warehouse"])[0] is resolve_sources(["warehouse"])[0]


def test_unknown_source_raises_listing_the_configured_ones(scratch_config, tmp_path):
    _setup(scratch_config, tmp_path / "w.db")
    from ingestlib.sources.registry import resolve_sources

    with pytest.raises(ValueError, match="unknown source 'ghost'"):
        resolve_sources(["ghost"])
    with pytest.raises(ValueError, match="warehouse"):  # names what IS configured
        resolve_sources(["ghost"])


def test_reset_registry_drops_the_cache(scratch_config, tmp_path):
    _setup(scratch_config, tmp_path / "w.db")
    from ingestlib.sources.registry import reset_registry, resolve_sources

    first = resolve_sources(["warehouse"])[0]
    reset_registry()
    assert resolve_sources(["warehouse"])[0] is not first
