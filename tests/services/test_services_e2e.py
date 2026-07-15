"""Full product round-trip: ingest a real document, retrieve with citations.

Opt-in via RUN_SERVICES_E2E=1 — needs the ENTIRE stack: VL server, Bedrock,
S3, the configured vector store, and Jina.
"""
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SERVICES_E2E") != "1",
    reason="services e2e is opt-in: set RUN_SERVICES_E2E=1 (needs full stack)",
)

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_PDF = _TESTS_DIR / "data" / "pdf" / "egov-survey.pdf"


@pytest.fixture(scope="module")
def ingested():
    from ingestlib.services import ingest

    result = ingest(_PDF, skip_existing=False)  # force the full pipeline
    time.sleep(8)  # serverless indexing is eventually consistent
    return result


def test_ingest_completes_all_stages(ingested):
    assert ingested.status == "ingested"
    assert ingested.vectors == ingested.chunks > 0
    assert set(ingested.durations) == {"parse", "classify", "split", "embed", "upsert"}


def test_reingest_is_skipped_by_checksum(ingested):
    from ingestlib.services import ingest

    again = ingest(_PDF)  # skip_existing defaults True
    assert again.status == "skipped"
    assert again.doc_id == ingested.doc_id


def test_retrieve_returns_cited_hits_from_this_doc(ingested):
    from ingestlib.services import retrieve

    result = retrieve("what do citizens think of e-government services?", top_k=3)
    ours = [h for h in result.hits if h.chunk.document_id == ingested.doc_id]
    assert ours, "expected at least one hit from the ingested document"
    top = ours[0]
    assert top.rerank_score is not None
    assert top.chunk.pages and top.citation.startswith("doc ")
    assert result.context  # prompt-ready block builds


def test_retrieve_filter_by_category(ingested):
    from ingestlib.services import retrieve

    result = retrieve("survey findings", top_k=5,
                      filters={"category": ingested.category})
    assert all(h.chunk.category == ingested.category for h in result.hits)


def test_retrieve_rejects_empty_question():
    from ingestlib.services import retrieve

    with pytest.raises(ValueError, match="non-empty"):
        retrieve("   ")
