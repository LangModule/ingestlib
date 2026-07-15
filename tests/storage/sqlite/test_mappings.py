"""FTS sanitization, RRF fusion, and payload round-trip — pure, always run."""
import pytest

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.sqlite.store import (
    _breadcrumb,
    _filter_sql,
    _from_payload,
    _fts_match,
    _rrf,
    _to_payload,
)


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=7, section="methods", heading="Recruitment", text="t", markdown="m",
        embedding_text="[research_paper › methods › Recruitment]\n\nm",
        pages=[4, 5], region_ids={4: [3, 4], 5: [0]}, kind="mixed", token_estimate=380,
    )


def test_fts_match_quotes_and_ors_word_tokens():
    assert _fts_match("recruited Cairo centers") == '"recruited" OR "Cairo" OR "centers"'


def test_fts_match_survives_query_syntax():
    # raw FTS5 operators and punctuation must come out as harmless quoted tokens
    assert _fts_match("what's the +360% growth?") == (
        '"what" OR "s" OR "the" OR "360" OR "growth"'
    )
    assert _fts_match('AND OR NOT "quoted" (grouped)*') == (
        '"AND" OR "OR" OR "NOT" OR "quoted" OR "grouped"'
    )


def test_fts_match_none_when_nothing_tokenizes():
    assert _fts_match("?!… — 🚀") is None
    assert _fts_match("   ") is None


def test_rrf_boosts_agreement_and_keeps_dense_on_ties():
    fused = _rrf([1, 2, 3], [3, 4])
    order = [rowid for rowid, _ in fused]
    assert order[0] == 3, "rowid on both lists must outrank any single-list rowid"
    assert order.index(1) < order.index(4), "equal-rank tie goes to the dense side"
    scores = dict(fused)
    assert scores[3] == pytest.approx(1 / 61 + 1 / 63)


def test_breadcrumb_skips_empty_parts():
    assert _breadcrumb(_chunk(), "research_paper") == "research_paper methods Recruitment"
    bare = _chunk().model_copy(update={"heading": ""})
    assert _breadcrumb(bare, "") == "methods"


def test_payload_round_trip_restores_types():
    payload = _to_payload("abc123", _chunk(), category="research_paper", namespace="")
    assert payload["region_ids"] == {"4": [3, 4], "5": [0]}  # JSON keys are strings
    hit = _from_payload(0.87, payload)
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]
    assert hit.region_ids == {4: [3, 4], 5: [0]}             # ints again
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_payload_defaults():
    hit = _from_payload(0.5, {"document_id": "d", "chunk_id": 1})
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"


def test_filter_sql_builds_prefixed_equality():
    sql, params = _filter_sql({"category": "invoice", "kind": "table"}, prefix="c.")
    assert sql == " AND c.category = ? AND c.kind = ?"
    assert params == ["invoice", "table"]
    assert _filter_sql(None) == ("", [])


def test_filter_sql_rejects_unknown_fields():
    with pytest.raises(ValueError, match="heading"):
        _filter_sql({"heading": "x"})
