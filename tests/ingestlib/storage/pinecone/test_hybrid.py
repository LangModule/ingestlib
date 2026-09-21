"""Hybrid merge logic — pure, always run."""
from ingestlib.storage.base import RetrievedChunk
from ingestlib.storage.pinecone.store import _merge_hits


def _hit(doc: str, cid: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(score=score, document_id=doc, chunk_id=cid)


def test_dense_order_is_preserved_and_sparse_appends():
    dense = [_hit("a", 0, 0.9), _hit("a", 1, 0.8)]
    sparse = [_hit("b", 0, 3.2)]
    merged = _merge_hits(dense, sparse)
    assert [(h.document_id, h.chunk_id) for h in merged] == [("a", 0), ("a", 1), ("b", 0)]


def test_duplicate_hits_keep_the_dense_version():
    dense = [_hit("a", 0, 0.9)]
    sparse = [_hit("a", 0, 3.2), _hit("a", 1, 2.1)]  # (a, 0) already found by dense
    merged = _merge_hits(dense, sparse)
    assert [(h.document_id, h.chunk_id) for h in merged] == [("a", 0), ("a", 1)]
    assert merged[0].score == 0.9  # the dense hit survives, not its sparse twin


def test_empty_sides_pass_through():
    dense = [_hit("a", 0)]
    assert _merge_hits(dense, []) == dense
    assert _merge_hits([], dense) == dense
    assert _merge_hits([], []) == []
