"""default_store() selection — pure, always run (constructors never connect)."""
import pytest

from ingestlib.storage import PineconeStore, QdrantStore, default_store


def test_current_config_selects_a_known_connector():
    store = default_store()
    assert isinstance(store, (PineconeStore, QdrantStore))


def test_unknown_name_raises_with_choices(monkeypatch):
    import ingestlib.config as config_module

    current = config_module.get_config()  # materialize the lazy singleton
    bad = current.__class__(**{**current.__dict__, "vector_store": "chroma"})
    monkeypatch.setattr(config_module, "_config", bad)
    with pytest.raises(ValueError, match="pinecone.*qdrant|qdrant.*pinecone"):
        default_store()
