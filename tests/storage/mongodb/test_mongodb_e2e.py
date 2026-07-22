"""Full connector round-trip against a real MongoDB with search support.

Opt-in via RUN_MONGODB_E2E=1 — needs a reachable deployment at MONGODB_URL
(Atlas any tier, docker run mongodb/mongodb-atlas-local, or self-managed
8.2+ with mongot). Embeddings are synthetic 8-dim vectors — no embedding
provider needed; the only requirement is the server. Uses a dedicated test collection
(dropped afterwards) so the user's real collection keeps its production
dimension. Search indexes are eventually consistent, so writes settle behind
a short wait before querying.
"""
import dataclasses
import os
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MONGODB_E2E") != "1",
    reason="mongodb e2e is opt-in: set RUN_MONGODB_E2E=1 (needs MongoDB at MONGODB_URL)",
)

_DOC_ID = "test-mongodb-doc"
_COLLECTION = "ingestlib_test_e2e"
_DIM = 8
_SYNC_SECONDS = 8  # search indexes lag writes


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
    from ingestlib.config import MongodbConfig, get_config
    from ingestlib.storage import MongodbStore
    from ingestlib.storage.mongodb.client import get_collection, reset_mongodb_client

    cfg = get_config()
    if not cfg.mongodb.url:
        pytest.skip("MONGODB_URL not set in .env")
    patched = dataclasses.replace(
        cfg,
        mongodb=MongodbConfig(
            url=cfg.mongodb.url,
            database=cfg.mongodb.database,
            collection_name=_COLLECTION,
        ),
    )
    config_module._config = patched
    reset_mongodb_client()
    yield MongodbStore()
    collection = get_collection()
    for idx in collection.list_search_indexes():
        collection.drop_search_index(idx["name"])
    collection.drop()
    config_module._config = cfg
    reset_mongodb_client()


@pytest.fixture(scope="module")
def upserted(store):
    n = store.upsert_chunks(_DOC_ID, _chunks(), _EMBEDDINGS, category="research_paper")
    time.sleep(_SYNC_SECONDS)
    return n


def test_upsert_writes_all_chunks(upserted):
    assert upserted == 2


def test_query_returns_nearest_chunk_with_full_payload(store, upserted):
    hits = store.query(_NEAR_RECRUITMENT, top_k=2)
    assert hits, "expected hits"
    assert hits[0].heading == "Participant recruitment"
    assert hits[0].document_id == _DOC_ID
    assert hits[0].region_ids == {4: [2, 3]}  # provenance round-trip
    assert hits[0].pages == [4]
    assert 0.0 < hits[0].score <= 1.0         # $vectorSearch score when dense-only


def test_query_filter_constrains_results(store, upserted):
    hits = store.query(_NEAR_REVENUE, top_k=5, filters={"section": "methods"})
    assert all(h.section == "methods" for h in hits)


def test_query_unknown_filter_field_raises(store, upserted):
    with pytest.raises(ValueError, match="heading"):
        store.query(_NEAR_REVENUE, top_k=5, filters={"heading": "Revenue growth"})


def test_namespaces_are_isolated(store, upserted):
    store.upsert_chunks("prod-doc", _chunks(), _EMBEDDINGS,
                        category="research_paper", namespace="prod")
    time.sleep(_SYNC_SECONDS)
    default_hits = store.query(_NEAR_RECRUITMENT, top_k=10)
    assert {h.document_id for h in default_hits} == {_DOC_ID}
    prod_hits = store.query(_NEAR_RECRUITMENT, top_k=10, namespace="prod")
    assert {h.document_id for h in prod_hits} == {"prod-doc"}
    assert store.delete_document("prod-doc", namespace="prod") == 2
    time.sleep(_SYNC_SECONDS)


def test_hybrid_fusion_surfaces_lexical_match(store, upserted):
    """RRF-fused query: exact tokens must surface the right chunk even when
    the dense vector is deliberately off-topic."""
    hits = store.query(_NEAR_REVENUE, top_k=2, text="recruited Cairo community centers")
    assert hits, "expected fused hits"
    assert any(h.heading == "Participant recruitment" for h in hits), (
        "BM25 branch should surface the Cairo chunk despite the off-topic dense vector"
    )


def test_lucene_stemming_matches_inflected_query(store, upserted):
    hits = store.query(_NEAR_REVENUE, top_k=2, text="recruiting cairo")
    assert any(h.heading == "Participant recruitment" for h in hits), (
        "'recruiting' must stem to match 'recruited'"
    )


def test_hybrid_query_has_no_duplicates(store, upserted):
    hits = store.query(_NEAR_RECRUITMENT, top_k=5, text="participants recruited Cairo")
    keys = [(h.document_id, h.chunk_id) for h in hits]
    assert len(keys) == len(set(keys)), "fusion must yield each chunk once"


def test_punctuation_heavy_text_is_a_search_not_an_error(store, upserted):
    # $search text is plain text, not a query language — nothing to sanitize
    hits = store.query(_NEAR_RECRUITMENT, top_k=2, text="what's the +360% growth?")
    assert hits
    hits = store.query(_NEAR_RECRUITMENT, top_k=2, text="?!… — 🚀")
    assert hits, "untokenizable text degrades to dense results"


def test_reupsert_with_fewer_chunks_leaves_no_orphans(store, upserted):
    """Delete-then-insert semantics: a re-parsed document that now has fewer
    chunks must not leave its old chunk_ids behind."""
    store.upsert_chunks(_DOC_ID, _chunks()[:1], _EMBEDDINGS[:1], category="research_paper")
    time.sleep(_SYNC_SECONDS)
    ours = [h for h in store.query(_NEAR_RECRUITMENT, top_k=10)
            if h.document_id == _DOC_ID]
    assert len(ours) == 1, "stale chunk_ids must be removed on re-ingestion"
    store.upsert_chunks(_DOC_ID, _chunks(), _EMBEDDINGS, category="research_paper")
    time.sleep(_SYNC_SECONDS)


def test_dimension_mismatch_raises(store, upserted):
    with pytest.raises(ValueError, match="8-dim"):
        store.query([1.0, 0.0, 0.0, 0.0], top_k=2)


def test_delete_document_removes_all_traces(store, upserted):
    assert store.delete_document(_DOC_ID) == 2
    time.sleep(_SYNC_SECONDS)
    assert store.query(_NEAR_RECRUITMENT, top_k=5) == []
    assert store.query(_NEAR_RECRUITMENT, top_k=5, text="recruited Cairo") == [], (
        "the lexical index must drop deleted documents once it syncs"
    )
    assert store.delete_document(_DOC_ID) == 0
