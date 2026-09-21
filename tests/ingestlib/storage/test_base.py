"""VectorStore contract behavior — pure, always run."""
import pytest

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore


def _chunk(cid: int = 0) -> Chunk:
    return Chunk(chunk_id=cid, section="s", text="t", markdown="m",
                 embedding_text="[s]\n\nm", pages=[1])


def test_abstract_contract_cannot_instantiate():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_retrieved_chunk_is_frozen_with_defaults():
    r = RetrievedChunk(score=0.9, document_id="d", chunk_id=1)
    assert r.region_ids == {} and r.pages == [] and r.kind == "text"
    with pytest.raises(Exception):
        r.score = 0.1  # type: ignore[misc]


def test_validate_upsert_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        VectorStore._validate_upsert([], [])


def test_validate_upsert_rejects_length_mismatch():
    with pytest.raises(ValueError, match="pair 1:1"):
        VectorStore._validate_upsert([_chunk()], [[0.1], [0.2]])


def test_validate_upsert_rejects_mixed_dimensions():
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        VectorStore._validate_upsert([_chunk(0), _chunk(1)], [[0.1, 0.2], [0.1]])


def test_validate_upsert_accepts_valid_pairing():
    VectorStore._validate_upsert([_chunk(0), _chunk(1)], [[0.1, 0.2], [0.3, 0.4]])
