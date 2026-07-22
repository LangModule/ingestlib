"""Full connector round-trip against real Pinecone + real embeddings from
the configured provider.

Opt-in via RUN_PINECONE_E2E=1 — creates the real indexes (dense + sparse) on
first ever run, upserts a sentinel document into both, queries dense, sparse,
and hybrid, then deletes it from both.
"""
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PINECONE_E2E") != "1",
    reason="pinecone e2e is opt-in: set RUN_PINECONE_E2E=1 "
           "(needs PINECONE_API_KEY + embedding-provider access)",
)

_DOC_ID = "e2e-test-pinecone-doc"


def _chunks():
    from ingestlib.operations.split.models import Chunk

    return [
        Chunk(chunk_id=0, section="methods", heading="Participant recruitment",
              text="Participants were recruited in Cairo.",
              markdown="Participants were recruited in Cairo.",
              embedding_text="[research_paper › methods › Participant recruitment]\n\n"
                             "Participants were recruited through community centers in Cairo.",
              pages=[4], region_ids={4: [2, 3]}),
        Chunk(chunk_id=1, section="results", heading="Revenue growth",
              text="Revenue grew 18% year over year to $82B.",
              markdown="Revenue grew 18% year over year to $82B.",
              embedding_text="[earnings_report › results › Revenue growth]\n\n"
                             "Revenue grew 18% year over year to $82B gross bookings.",
              pages=[2], region_ids={2: [7]}),
    ]


@pytest.fixture(scope="module")
def store():
    from ingestlib.config import get_pinecone_config
    from ingestlib.storage import PineconeStore

    if not get_pinecone_config().api_key:
        pytest.skip("PINECONE_API_KEY not set in .env")
    s = PineconeStore()
    yield s
    s.delete_document(_DOC_ID)


@pytest.fixture(scope="module")
def upserted(store):
    from ingestlib.foundations.llm import embed_text

    chunks = _chunks()
    embeddings = [embed_text(c.embedding_text) for c in chunks]
    n = store.upsert_chunks(_DOC_ID, chunks, embeddings, category="research_paper")
    # serverless indexing is eventually consistent — give it a moment
    time.sleep(8)
    return n


def test_upsert_writes_all_vectors(upserted):
    assert upserted == 2


def test_query_returns_semantically_right_chunk(store, upserted):
    from ingestlib.foundations.llm import embed_text

    q = embed_text("how were study participants recruited?", purpose="GENERIC_RETRIEVAL")
    hits = store.query(q, top_k=2)
    assert hits, "expected hits"
    assert hits[0].heading == "Participant recruitment"
    assert hits[0].document_id == _DOC_ID
    assert hits[0].region_ids == {4: [2, 3]}  # provenance round-trip


def test_query_filter_constrains_results(store, upserted):
    from ingestlib.foundations.llm import embed_text

    q = embed_text("revenue growth", purpose="GENERIC_RETRIEVAL")
    hits = store.query(q, top_k=5, filters={"section": "methods"})
    assert all(h.section == "methods" for h in hits)


def test_sparse_index_finds_lexical_match(store, upserted):
    """The sparse half alone must find the chunk by its distinctive tokens."""
    hits = store._query_sparse(
        "recruited Cairo community centers", top_k=2, filters=None, namespace=""
    )
    assert hits, "expected sparse hits"
    assert hits[0].document_id == _DOC_ID and hits[0].chunk_id == 0


def test_hybrid_query_merges_without_duplicates(store, upserted):
    from ingestlib.foundations.llm import embed_text

    q = embed_text("how were study participants recruited?", purpose="GENERIC_RETRIEVAL")
    hits = store.query(q, top_k=2, text="how were study participants recruited?")
    keys = [(h.document_id, h.chunk_id) for h in hits]
    assert len(keys) == len(set(keys)), "a chunk found by both sides must appear once"
    assert any(h.heading == "Participant recruitment" for h in hits)


def test_reupsert_with_fewer_chunks_leaves_no_orphans(store, upserted):
    """A re-parsed document that now has fewer chunks must not leave its old
    chunk_ids behind as stale vectors in either index."""
    from ingestlib.foundations.llm import embed_text

    chunks = _chunks()
    embeddings = [embed_text(c.embedding_text) for c in chunks]
    store.upsert_chunks(_DOC_ID, chunks[:1], embeddings[:1], category="research_paper")
    time.sleep(8)  # eventual consistency
    q = embed_text("participants recruited revenue growth", purpose="GENERIC_RETRIEVAL")
    ours = [h for h in store.query(q, top_k=10) if h.document_id == _DOC_ID]
    assert len(ours) == 1, "stale chunk_ids must be pruned on re-ingestion"
    sparse = store._query_sparse("revenue grew year over year", 10, None, "")
    assert [h for h in sparse if h.document_id == _DOC_ID and h.chunk_id == 1] == [], (
        "the sparse index must be pruned too"
    )
    store.upsert_chunks(_DOC_ID, chunks, embeddings, category="research_paper")
    time.sleep(8)


def test_delete_document_removes_vectors_from_both_indexes(store, upserted):
    deleted = store.delete_document(_DOC_ID)
    assert deleted == 2
    time.sleep(5)
    from ingestlib.foundations.llm import embed_text

    q = embed_text("participants recruited", purpose="GENERIC_RETRIEVAL")
    hits = [h for h in store.query(q, top_k=5) if h.document_id == _DOC_ID]
    assert hits == []
    sparse = store._query_sparse("recruited Cairo community centers", 5, None, "")
    assert [h for h in sparse if h.document_id == _DOC_ID] == []
