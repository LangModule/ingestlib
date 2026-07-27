"""backfill() — a vector store rebuilt from stored artifacts. Always run:
embedding is stubbed at the ingestor seam (backfill reuses _embed_chunks),
artifacts + sqlite are real, so the rebuild path is proven end to end.
"""
import dataclasses

import pytest

import ingestlib.config as config_module
from ingestlib.config import SqliteConfig, get_config
from ingestlib.services import backfill
from ingestlib.storage import SqliteStore

from tests.services.conftest import vec

_DOC_A = "backfill-a-" + "0" * 53
_DOC_B = "backfill-b-" + "1" * 53


@pytest.fixture()
def stub_embeddings(monkeypatch):
    """backfill embeds through the ingestor's _embed_chunks → aembed_text."""
    import importlib

    ingestor = importlib.import_module("ingestlib.services.ingest.ingestor")

    async def fake_embed(text, purpose="GENERIC_INDEX", dimension=1024):
        return vec(1.0)

    monkeypatch.setattr(ingestor, "aembed_text", fake_embed)


def test_backfill_rebuilds_a_wiped_store(stack, make_document, stub_embeddings,
                                         tmp_path, monkeypatch):
    """The provider-switch / new-store scenario: artifacts exist, the target
    store is empty — backfill fills it without any pipeline run."""
    make_document(_DOC_A, stack.corpus / "a.pdf")
    make_document(_DOC_B, stack.corpus / "b.pdf")

    # a FRESH store standing in for "the new connector"
    cfg = dataclasses.replace(
        get_config(), sqlite=SqliteConfig(path=tmp_path / "fresh.db")
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    fresh = SqliteStore()
    assert fresh.query(vec(1.0), top_k=5) == []

    result = backfill(store=fresh)

    assert result.documents == 2
    assert result.chunks == 2
    assert result.skipped == []
    hits = fresh.query(vec(1.0), top_k=5)
    assert {h.document_id for h in hits} == {_DOC_A, _DOC_B}


def test_backfill_skips_docs_without_split_artifacts(stack, make_document,
                                                     stub_embeddings):
    make_document(_DOC_A, stack.corpus / "a.pdf")
    make_document(_DOC_B, stack.corpus / "b.pdf", with_vectors=False)  # parse only

    result = backfill(store=stack.store)

    assert result.documents == 1
    assert result.skipped == [_DOC_B]


def test_backfill_is_namespace_scoped(stack, make_document, stub_embeddings):
    make_document(_DOC_A, stack.corpus / "a.pdf")
    make_document(_DOC_B, stack.corpus / "b.pdf", namespace="tenant-a")

    result = backfill(store=stack.store, namespace="tenant-a")

    assert result.documents == 1
    hits = stack.store.query(vec(1.0), top_k=5, namespace="tenant-a")
    assert {h.document_id for h in hits} == {_DOC_B}


def test_backfill_is_idempotent(stack, make_document, stub_embeddings):
    make_document(_DOC_A, stack.corpus / "a.pdf")
    backfill(store=stack.store)
    result = backfill(store=stack.store)  # again — upserts overwrite
    assert result.documents == 1
    hits = stack.store.query(vec(1.0), top_k=10)
    assert len([h for h in hits if h.document_id == _DOC_A]) == 1, "no duplicates"


def test_empty_registry_backfills_nothing(stack, stub_embeddings):
    result = backfill(store=stack.store)
    assert result.documents == 0 and result.chunks == 0
