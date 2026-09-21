"""Registry-backed retrieve filters — collection + min_confidence.

Uses the `pipeline` fixture (stubbed embeddings return one vector, so both docs
match every query equally; the filters are what separate them).
"""
import pytest

from tests.ingestlib.services.conftest import vec


@pytest.fixture(autouse=True)
def _stub_retrieve_embed(monkeypatch):
    """The pipeline fixture stubs the INGESTOR's embed; the retriever imports its
    own aembed_text — stub it too, to the same 8-dim vector the store holds."""
    from ingestlib.services.retrieve import retriever

    async def fake_embed(text, purpose="GENERIC_RETRIEVAL", dimension=1024):
        return vec(1.0)

    monkeypatch.setattr(retriever, "aembed_text", fake_embed)


def _ingest_two(pipeline):
    from ingestlib.services import ingest
    from ingestlib.utils.files import sha256_of_file

    a = pipeline.corpus / "a.pdf"
    a.write_bytes(b"alpha doc about revenue")
    b = pipeline.corpus / "b.pdf"
    b.write_bytes(b"beta doc about revenue")
    ingest(a, store=pipeline.store)
    ingest(b, store=pipeline.store)
    return sha256_of_file(a), sha256_of_file(b)


def test_min_confidence_filters_low_confidence_docs(pipeline):
    from ingestlib.services import retrieve
    from ingestlib_registry.db import session_scope
    from ingestlib_registry.models import Document

    a_id, b_id = _ingest_two(pipeline)
    with session_scope() as s:  # drop doc a's confidence below the threshold
        s.get(Document, a_id).classify_confidence = 0.2

    filtered = retrieve("revenue", store=pipeline.store, rerank=False, min_confidence=0.5)
    ids = {h.chunk.document_id for h in filtered.hits}
    assert b_id in ids and a_id not in ids

    unfiltered = retrieve("revenue", store=pipeline.store, rerank=False)
    assert {a_id, b_id} <= {h.chunk.document_id for h in unfiltered.hits}


def test_collection_filter(pipeline):
    from ingestlib.services import retrieve
    from ingestlib_registry.db import session_scope
    from ingestlib_registry.models import Document

    a_id, b_id = _ingest_two(pipeline)
    with session_scope() as s:  # put only doc a in the 'finance' collection
        s.get(Document, a_id).collection = "finance"

    result = retrieve("revenue", store=pipeline.store, rerank=False, collection="finance")
    ids = {h.chunk.document_id for h in result.hits}
    assert a_id in ids and b_id not in ids


def test_hits_carry_content_and_registry_enrichment(pipeline):
    from ingestlib.services import retrieve

    _ingest_two(pipeline)
    # the vector store keeps the chunk body (fat payload) — retrieve stays
    # vector-store-driven for content, no registry round-trip to read it
    raw = pipeline.store.query(vec(1.0), top_k=5)
    assert raw and all(c.markdown == "hello" for c in raw)

    # retrieve enriches each hit with the doc's collection (classify wrote it)
    result = retrieve("revenue", store=pipeline.store, rerank=False)
    assert result.hits
    assert all(h.chunk.markdown == "hello" for h in result.hits)   # content from the store
    assert all(h.collection == "report" for h in result.hits)      # enriched from the registry
    assert result.context.strip()                                  # prompt block builds
