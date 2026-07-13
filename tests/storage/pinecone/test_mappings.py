"""Metadata flatten/unflatten and ID mapping — pure, always run."""
from ingestlib.operations.split.models import Chunk
from ingestlib.storage.pinecone.store import _from_match, _to_metadata, _vector_id


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=7, section="methods", heading="Recruitment", text="t", markdown="m",
        embedding_text="[research_paper › methods › Recruitment]\n\nm",
        pages=[4, 5], region_ids={4: [3, 4], 5: [0]}, kind="mixed", token_estimate=380,
    )


def test_vector_id_is_doc_colon_chunk():
    assert _vector_id("abc123", 7) == "abc123:7"


def test_metadata_is_flat_and_json_safe():
    md = _to_metadata("abc123", _chunk(), category="research_paper")
    assert md["pages"] == ["4", "5"]                       # list of strings
    assert isinstance(md["region_ids"], str)               # JSON-encoded
    assert md["category"] == "research_paper"
    for v in md.values():
        assert isinstance(v, (str, int, float, bool, list))


def test_round_trip_restores_types():
    md = _to_metadata("abc123", _chunk(), category="research_paper")
    hit = _from_match({"score": 0.87, "metadata": md})
    assert hit.score == 0.87
    assert hit.document_id == "abc123" and hit.chunk_id == 7
    assert hit.pages == [4, 5]                             # ints again
    assert hit.region_ids == {4: [3, 4], 5: [0]}           # dict[int, list[int]] again
    assert hit.section == "methods" and hit.kind == "mixed"


def test_missing_optional_metadata_defaults():
    hit = _from_match({"score": 0.5, "metadata": {"document_id": "d", "chunk_id": 1}})
    assert hit.pages == [] and hit.region_ids == {} and hit.kind == "text"
