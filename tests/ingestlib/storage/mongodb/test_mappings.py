"""Chunk-key mapping, RRF fusion, and payload round-trip — pure, always run."""
import pytest

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.mongodb.store import (
    _breadcrumb,
    _check_filters,
    _chunk_key,
    _from_payload,
    _rrf,
    _to_document,
)


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=7, section="methods", heading="Recruitment", text="t", markdown="m",
        embedding_text="[research_paper › methods › Recruitment]\n\nm",
        pages=[4, 5], region_ids={4: [3, 4], 5: [0]}, kind="mixed", token_estimate=380,
    )


def test_chunk_key_is_deterministic_and_namespace_scoped():
    assert _chunk_key("", "abc123", 7) == ":abc123:7"
    assert _chunk_key("prod", "abc123", 7) == "prod:abc123:7"
    assert _chunk_key("", "abc123", 7) != _chunk_key("prod", "abc123", 7)


def test_rrf_boosts_agreement_and_keeps_dense_on_ties():
    fused = _rrf(["a", "b", "c"], ["c", "d"])
    order = [doc_id for doc_id, _ in fused]
    assert order[0] == "c", "id on both lists must outrank any single-list id"
    assert order.index("a") < order.index("d"), "equal-rank tie goes to the dense side"
    scores = dict(fused)
    assert scores["c"] == pytest.approx(1 / 61 + 1 / 63)


def test_breadcrumb_skips_empty_parts():
    assert _breadcrumb(_chunk(), "research_paper") == "research_paper methods Recruitment"
    bare = _chunk().model_copy(update={"heading": ""})
    assert _breadcrumb(bare, "") == "methods"


def test_document_shape_and_payload_round_trip():
    doc = _to_document("abc123", _chunk(), [0.1, 0.2], "research_paper", namespace="")
    assert doc["_id"] == ":abc123:7"
    assert doc["embedding"] == [0.1, 0.2]
    assert doc["breadcrumb"] == "research_paper methods Recruitment"
    assert doc["payload"]["region_ids"] == {"4": [3, 4], "5": [0]}  # BSON keys are strings
    hit = _from_payload(0.87, doc["payload"])
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]
    assert hit.region_ids == {4: [3, 4], 5: [0]}                    # ints again
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_payload_defaults():
    hit = _from_payload(0.5, {"document_id": "d", "chunk_id": 1})
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"


def test_check_filters_rejects_unknown_fields():
    _check_filters(None)
    _check_filters({"category": "invoice", "kind": "table"})
    with pytest.raises(ValueError, match="heading"):
        _check_filters({"heading": "x"})
