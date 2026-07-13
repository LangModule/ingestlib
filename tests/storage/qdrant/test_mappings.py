"""Point-ID mapping and payload round-trip — pure, always run."""
import uuid
from types import SimpleNamespace

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.qdrant.store import _filter, _from_point, _point_id, _to_payload


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=7, section="methods", heading="Recruitment", text="t", markdown="m",
        embedding_text="[research_paper › methods › Recruitment]\n\nm",
        pages=[4, 5], region_ids={4: [3, 4], 5: [0]}, kind="mixed", token_estimate=380,
    )


def test_point_id_is_deterministic_valid_uuid():
    a = _point_id("abc123", 7)
    b = _point_id("abc123", 7)
    assert a == b, "same doc:chunk must always map to the same point ID"
    assert uuid.UUID(a)  # parses as a real UUID
    assert _point_id("abc123", 8) != a
    assert _point_id("other", 7) != a


def test_payload_round_trip_restores_types():
    payload = _to_payload("abc123", _chunk(), category="research_paper", namespace="")
    assert payload["region_ids"] == {"4": [3, 4], "5": [0]}  # JSON keys are strings
    hit = _from_point(SimpleNamespace(score=0.87, payload=payload))
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]
    assert hit.region_ids == {4: [3, 4], 5: [0]}             # ints again
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_payload_defaults():
    hit = _from_point(
        SimpleNamespace(score=0.5, payload={"document_id": "d", "chunk_id": 1})
    )
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"


def test_filter_always_pins_namespace():
    f = _filter("prod", filters={"category": "invoice"}, document_id="d1")
    keys = [c.key for c in f.must]
    assert keys == ["namespace", "category", "document_id"]
