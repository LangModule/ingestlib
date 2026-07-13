"""Full connector round-trip against a real Qdrant server + real Nova embeddings.

Opt-in via RUN_QDRANT_E2E=1 — needs a reachable server at QDRANT_URL
(e.g. docker run -p 6333:6333 qdrant/qdrant, or Qdrant Cloud with an API key).
Creates the real collection on first run, upserts a sentinel document (dense +
BM25 sparse), queries dense, hybrid-fused, and lexical, then deletes it.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QDRANT_E2E") != "1",
    reason="qdrant e2e is opt-in: set RUN_QDRANT_E2E=1 (needs a Qdrant server + Bedrock)",
)

_DOC_ID = "e2e-test-qdrant-doc"


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
    from ingestlib.storage import QdrantStore

    s = QdrantStore()
    yield s
    s.delete_document(_DOC_ID)


@pytest.fixture(scope="module")
def upserted(store):
    from ingestlib.foundations.llm import embed_text

    chunks = _chunks()
    embeddings = [embed_text(c.embedding_text) for c in chunks]
    return store.upsert_chunks(_DOC_ID, chunks, embeddings, category="research_paper")


def test_upsert_writes_all_points(upserted):
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


def test_hybrid_fusion_surfaces_lexical_match(store, upserted):
    """RRF-fused query: exact tokens must surface the right chunk even when
    the dense vector is deliberately off-topic."""
    from ingestlib.foundations.llm import embed_text

    off_topic = embed_text("quarterly financial performance", purpose="GENERIC_RETRIEVAL")
    hits = store.query(off_topic, top_k=2, text="recruited Cairo community centers")
    assert hits, "expected fused hits"
    assert any(h.heading == "Participant recruitment" for h in hits), (
        "BM25 branch should surface the Cairo chunk despite the off-topic dense vector"
    )


def test_hybrid_query_has_no_duplicates(store, upserted):
    from ingestlib.foundations.llm import embed_text

    q = embed_text("how were study participants recruited?", purpose="GENERIC_RETRIEVAL")
    hits = store.query(q, top_k=5, text="how were study participants recruited?")
    keys = [(h.document_id, h.chunk_id) for h in hits]
    assert len(keys) == len(set(keys)), "fusion must yield each chunk once"


def test_reupsert_overwrites_not_duplicates(store, upserted):
    from ingestlib.foundations.llm import embed_text

    chunks = _chunks()
    embeddings = [embed_text(c.embedding_text) for c in chunks]
    store.upsert_chunks(_DOC_ID, chunks, embeddings, category="research_paper")
    q = embed_text("participants recruited", purpose="GENERIC_RETRIEVAL")
    ours = [h for h in store.query(q, top_k=10) if h.document_id == _DOC_ID]
    assert len(ours) == 2, "deterministic point IDs must overwrite, never duplicate"


def test_delete_document_removes_points(store, upserted):
    deleted = store.delete_document(_DOC_ID)
    assert deleted == 2
    from ingestlib.foundations.llm import embed_text

    q = embed_text("participants recruited", purpose="GENERIC_RETRIEVAL")
    hits = [h for h in store.query(q, top_k=5) if h.document_id == _DOC_ID]
    assert hits == []
