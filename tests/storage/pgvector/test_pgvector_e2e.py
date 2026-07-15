"""Full connector round-trip against a real Postgres server with pgvector.

Opt-in via RUN_PGVECTOR_E2E=1 — needs a reachable server at PGVECTOR_URL
(e.g. docker run -p 5433:5432 pgvector/pgvector:pg18, or any managed
Postgres that ships the extension). Embeddings are synthetic 8-dim vectors —
the store contract takes vectors, so no Bedrock is needed; the only
requirement is the server. Uses a dedicated test table (dropped afterwards)
so the user's real table keeps its production dimension.
"""
import dataclasses
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PGVECTOR_E2E") != "1",
    reason="pgvector e2e is opt-in: set RUN_PGVECTOR_E2E=1 (needs Postgres at PGVECTOR_URL)",
)

_DOC_ID = "test-pgvector-doc"
_TABLE = "ingestlib_test_e2e"
_DIM = 8


def _vec(*values: float) -> list[float]:
    return list(values) + [0.0] * (_DIM - len(values))


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


_EMBEDDINGS = [_vec(1.0), _vec(0.0, 1.0)]          # chunk 0 → e1, chunk 1 → e2
_NEAR_RECRUITMENT = _vec(0.9, 0.1)
_NEAR_REVENUE = _vec(0.1, 0.9)


@pytest.fixture(scope="module")
def store():
    import ingestlib.config as config_module
    from ingestlib.config import PgvectorConfig, get_config
    from ingestlib.storage import PgvectorStore
    from ingestlib.storage.pgvector.client import connect, reset_pgvector

    cfg = get_config()
    if not cfg.pgvector.url:
        pytest.skip("PGVECTOR_URL not set in .env")
    patched = dataclasses.replace(
        cfg, pgvector=PgvectorConfig(url=cfg.pgvector.url, table_name=_TABLE)
    )
    config_module._config = patched
    reset_pgvector()
    yield PgvectorStore()
    with connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {_TABLE}")
    config_module._config = cfg
    reset_pgvector()


@pytest.fixture(scope="module")
def upserted(store):
    return store.upsert_chunks(_DOC_ID, _chunks(), _EMBEDDINGS, category="research_paper")


def test_upsert_writes_all_chunks(upserted):
    assert upserted == 2


def test_query_returns_nearest_chunk_with_full_payload(store, upserted):
    hits = store.query(_NEAR_RECRUITMENT, top_k=2)
    assert hits, "expected hits"
    assert hits[0].heading == "Participant recruitment"
    assert hits[0].document_id == _DOC_ID
    assert hits[0].region_ids == {4: [2, 3]}  # provenance round-trip
    assert hits[0].pages == [4]
    assert 0.0 < hits[0].score <= 1.0         # cosine similarity when dense-only


def test_query_filter_constrains_results(store, upserted):
    hits = store.query(_NEAR_REVENUE, top_k=5, filters={"section": "methods"})
    assert all(h.section == "methods" for h in hits)


def test_query_unknown_filter_field_raises(store, upserted):
    with pytest.raises(ValueError, match="heading"):
        store.query(_NEAR_REVENUE, top_k=5, filters={"heading": "Revenue growth"})


def test_namespaces_are_isolated(store, upserted):
    store.upsert_chunks("prod-doc", _chunks(), _EMBEDDINGS,
                        category="research_paper", namespace="prod")
    default_hits = store.query(_NEAR_RECRUITMENT, top_k=10)
    assert {h.document_id for h in default_hits} == {_DOC_ID}
    prod_hits = store.query(_NEAR_RECRUITMENT, top_k=10, namespace="prod")
    assert {h.document_id for h in prod_hits} == {"prod-doc"}
    assert store.delete_document("prod-doc", namespace="prod") == 2


def test_hybrid_fusion_surfaces_lexical_match(store, upserted):
    """RRF-fused query: exact tokens must surface the right chunk even when
    the dense vector is deliberately off-topic."""
    hits = store.query(_NEAR_REVENUE, top_k=2, text="recruited Cairo community centers")
    assert hits, "expected fused hits"
    assert any(h.heading == "Participant recruitment" for h in hits), (
        "full-text branch should surface the Cairo chunk despite the off-topic dense vector"
    )


def test_english_stemming_matches_inflected_query(store, upserted):
    hits = store.query(_NEAR_REVENUE, top_k=2, text="recruiting cairo")
    assert any(h.heading == "Participant recruitment" for h in hits), (
        "'recruiting' must stem to match 'recruited'"
    )


def test_hybrid_query_has_no_duplicates(store, upserted):
    hits = store.query(_NEAR_RECRUITMENT, top_k=5, text="participants recruited Cairo")
    keys = [(h.document_id, h.chunk_id) for h in hits]
    assert len(keys) == len(set(keys)), "fusion must yield each chunk once"


def test_query_syntax_in_text_degrades_cleanly(store, upserted):
    # raw tsquery operators/punctuation must never crash a hybrid query
    hits = store.query(_NEAR_RECRUITMENT, top_k=2, text="what's the +360% growth?")
    assert hits
    hits = store.query(_NEAR_RECRUITMENT, top_k=2, text="?!… — 🚀")
    assert hits, "untokenizable text falls back to dense-only"


def test_reupsert_with_fewer_chunks_leaves_no_orphans(store, upserted):
    """Delete-then-insert semantics: a re-parsed document that now has fewer
    chunks must not leave its old chunk_ids behind."""
    store.upsert_chunks(_DOC_ID, _chunks()[:1], _EMBEDDINGS[:1], category="research_paper")
    ours = [h for h in store.query(_NEAR_RECRUITMENT, top_k=10)
            if h.document_id == _DOC_ID]
    assert len(ours) == 1, "stale chunk_ids must be removed on re-ingestion"
    store.upsert_chunks(_DOC_ID, _chunks(), _EMBEDDINGS, category="research_paper")


def test_dimension_mismatch_raises(store, upserted):
    with pytest.raises(ValueError, match="8-dim"):
        store.query([1.0, 0.0, 0.0, 0.0], top_k=2)


def test_delete_document_removes_all_traces(store, upserted):
    assert store.delete_document(_DOC_ID) == 2
    assert store.query(_NEAR_RECRUITMENT, top_k=5) == []
    assert store.query(_NEAR_RECRUITMENT, top_k=5, text="recruited Cairo") == [], (
        "the lexical side must vanish with the rows (generated column)"
    )
    assert store.delete_document(_DOC_ID) == 0
