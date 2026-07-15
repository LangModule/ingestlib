"""default_store() selection — pure, always run (constructors never connect)."""
import pytest

import ingestlib.config as config_module
from ingestlib.storage import (
    MongodbStore,
    PgvectorStore,
    PineconeStore,
    QdrantStore,
    SqliteStore,
    default_store,
)


def _with_store(monkeypatch, name: str) -> None:
    current = config_module.get_config()  # materialize the lazy singleton
    patched = current.__class__(**{**current.__dict__, "vector_store": name})
    monkeypatch.setattr(config_module, "_config", patched)


def test_current_config_selects_a_known_connector():
    store = default_store()
    assert isinstance(
        store, (PineconeStore, QdrantStore, SqliteStore, PgvectorStore, MongodbStore)
    )


@pytest.mark.parametrize(("name", "cls"), [
    ("pinecone", PineconeStore),
    ("qdrant", QdrantStore),
    ("sqlite", SqliteStore),
    ("pgvector", PgvectorStore),
    ("mongodb", MongodbStore),
])
def test_each_name_selects_its_connector(monkeypatch, name, cls):
    _with_store(monkeypatch, name)
    assert isinstance(default_store(), cls)


def test_unknown_name_raises_with_choices(monkeypatch):
    _with_store(monkeypatch, "chroma")
    with pytest.raises(ValueError, match="mongodb.*pgvector.*pinecone.*qdrant.*sqlite"):
        default_store()
