"""Document-key mapping and payload round-trip — pure, always run."""
import pytest

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.opensearch.store import (
    _chunk_key,
    _filter_terms,
    _from_payload,
    _to_document,
)


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=7, section="methods", heading="Recruitment", text="t", markdown="m",
        embedding_text="[research_paper › methods › Recruitment]\n\nm",
        pages=[4, 5], region_ids={4: [3, 4], 5: [0]}, kind="mixed", token_estimate=380,
    )


def test_chunk_key_is_deterministic_and_namespace_scoped():
    default = _chunk_key("", "abc123", 7)
    assert _chunk_key("", "abc123", 7) == default
    assert _chunk_key("prod", "abc123", 7) != default, (
        "namespaces must never collide on the same chunk"
    )
    assert _chunk_key("", "abc123", 8) != default
    assert _chunk_key("", "other", 7) != default


def test_document_round_trip_restores_types():
    doc = _to_document("abc123", _chunk(), [0.1, 0.2], category="research_paper", namespace="")
    assert doc["breadcrumb"] == "research_paper methods Recruitment"
    assert doc["payload"]["region_ids"] == {"4": [3, 4], "5": [0]}  # JSON keys are strings
    hit = _from_payload(0.87, doc["payload"])
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]
    assert hit.region_ids == {4: [3, 4], 5: [0]}                    # ints again
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_payload_defaults():
    hit = _from_payload(0.5, {"document_id": "d", "chunk_id": 1})
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"


def test_filter_terms_always_pin_namespace():
    terms = _filter_terms("prod", filters={"category": "invoice"}, document_id="d1")
    keys = [next(iter(t["term"])) for t in terms]
    assert keys == ["namespace", "category", "document_id"]


def test_unknown_filter_field_raises():
    with pytest.raises(ValueError, match="unsupported filter field"):
        _filter_terms("", filters={"heading": "x"})
