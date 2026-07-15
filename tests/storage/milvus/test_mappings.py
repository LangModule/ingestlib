"""Filter-expression building, row shape, and payload round-trip — pure, always run."""
import pytest

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.milvus.store import (
    _breadcrumb,
    _chunk_key,
    _escape,
    _expr,
    _from_hit,
    _to_row,
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


def test_expr_always_pins_namespace_and_quotes_values():
    assert _expr("") == 'namespace == ""'
    assert _expr("prod", {"category": "invoice"}, document_id="d1") == (
        'namespace == "prod" and category == "invoice" and document_id == "d1"'
    )


def test_expr_escapes_quotes_and_backslashes():
    assert _escape('a"b\\c') == 'a\\"b\\\\c'
    expr = _expr("", {"section": 'x" or namespace != "'})
    assert '\\"' in expr, "quote characters in values must not break out of the string"


def test_expr_rejects_unknown_fields():
    with pytest.raises(ValueError, match="heading"):
        _expr("", {"heading": "x"})


def test_breadcrumb_skips_empty_parts():
    assert _breadcrumb(_chunk(), "research_paper") == "research_paper methods Recruitment"
    bare = _chunk().model_copy(update={"heading": ""})
    assert _breadcrumb(bare, "") == "methods"


def test_row_shape_and_payload_round_trip():
    row = _to_row("abc123", _chunk(), [0.1, 0.2], "research_paper", namespace="")
    assert row["id"] == ":abc123:7"
    assert row["embedding"] == [0.1, 0.2]
    assert row["search_text"].startswith("research_paper methods Recruitment\n")
    assert "sparse" not in row, "sparse vectors are computed server-side"
    assert row["payload"]["region_ids"] == {"4": [3, 4], "5": [0]}  # JSON keys are strings
    hit = _from_hit({"distance": 0.87, "entity": {"payload": row["payload"]}})
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]
    assert hit.region_ids == {4: [3, 4], 5: [0]}                    # ints again
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_payload_defaults():
    hit = _from_hit({"distance": 0.5, "entity": {"payload": {"document_id": "d", "chunk_id": 1}}})
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"
