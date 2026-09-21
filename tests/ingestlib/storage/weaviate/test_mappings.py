"""Object-ID mapping and payload round-trip — pure, always run."""
import uuid

import pytest

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.weaviate.store import (
    _filter,
    _from_payload,
    _object_id,
    _to_properties,
)


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=7, section="methods", heading="Recruitment", text="t", markdown="m",
        embedding_text="[research_paper › methods › Recruitment]\n\nm",
        pages=[4, 5], region_ids={4: [3, 4], 5: [0]}, kind="mixed", token_estimate=380,
    )


def test_object_id_is_deterministic_valid_uuid():
    a = _object_id("abc123", 7)
    b = _object_id("abc123", 7)
    assert a == b, "same doc:chunk must always map to the same object ID"
    assert uuid.UUID(a)  # parses as a real UUID
    assert _object_id("abc123", 8) != a
    assert _object_id("other", 7) != a


def test_object_id_is_namespace_scoped():
    default = _object_id("abc123", 7)
    assert _object_id("abc123", 7, namespace="") == default
    prod = _object_id("abc123", 7, namespace="prod")
    assert prod != default, "namespaces must never collide on the same chunk"
    assert _object_id("abc123", 7, namespace="prod") == prod  # still deterministic


def test_properties_round_trip_restores_types():
    props = _to_properties("abc123", _chunk(), category="research_paper", namespace="")
    assert props["breadcrumb"] == "research_paper methods Recruitment"
    hit = _from_payload(0.87, props["payload"])
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]
    assert hit.region_ids == {4: [3, 4], 5: [0]}    # ints again after the JSON trip
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_payload_defaults():
    hit = _from_payload(0.5, '{"document_id": "d", "chunk_id": 1}')
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"


def test_filter_builds_without_error():
    assert _filter("prod", filters={"category": "invoice"}, document_id="d1") is not None


def test_unknown_filter_field_raises():
    with pytest.raises(ValueError, match="unsupported filter field"):
        _filter("", filters={"heading": "x"})
