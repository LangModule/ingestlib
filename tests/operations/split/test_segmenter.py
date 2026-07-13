"""Boundary-repair and guarantee logic — pure, always run."""
from ingestlib.operations.split.pages import Block
from ingestlib.operations.split.segmenter import (
    _ChunkSpan,
    _enforce_heading_binding,
    _enforce_max_tokens,
    _merge_micro_chunks,
    _spans_to_groups,
)


def _block(kind: str = "text", chars: int = 400, page: int = 1) -> Block:
    return Block(page_num=page, kind=kind, markdown="x" * chars, text="x" * chars, region_ids=())


def test_spans_full_coverage_passthrough():
    spans = [_ChunkSpan(start_block=0, end_block=1, heading="a"),
             _ChunkSpan(start_block=2, end_block=3, heading="b")]
    groups = _spans_to_groups(spans, 4)
    assert [g[0] for g in groups] == [[0, 1], [2, 3]]


def test_spans_gap_is_repaired():
    spans = [_ChunkSpan(start_block=0, end_block=0, heading="a"),
             _ChunkSpan(start_block=3, end_block=4, heading="b")]  # blocks 1-2 skipped
    groups = _spans_to_groups(spans, 5)
    covered = sorted(i for g in groups for i in g[0])
    assert covered == [0, 1, 2, 3, 4]


def test_spans_overlap_is_repaired():
    spans = [_ChunkSpan(start_block=0, end_block=2, heading="a"),
             _ChunkSpan(start_block=1, end_block=4, heading="b")]  # overlap 1-2
    groups = _spans_to_groups(spans, 5)
    covered = sorted(i for g in groups for i in g[0])
    assert covered == [0, 1, 2, 3, 4]
    assert len(covered) == len(set(covered)), "no block in two chunks"


def test_spans_uncovered_tail_lands_in_final_group():
    groups = _spans_to_groups([_ChunkSpan(start_block=0, end_block=1, heading="a")], 4)
    assert sorted(i for g in groups for i in g[0]) == [0, 1, 2, 3]


def test_heading_never_ends_a_chunk():
    blocks = [_block(), _block(kind="heading"), _block(), _block()]
    groups = [([0, 1], "first"), ([2, 3], "second")]
    fixed = _enforce_heading_binding(groups, blocks)
    assert fixed[0][0] == [0]
    assert fixed[1][0] == [1, 2, 3], "heading moved down to bind with its content"


def test_oversized_chunk_splits_at_block_boundaries():
    blocks = [_block(chars=2000), _block(chars=2000), _block(chars=2000)]  # 500 tok each
    groups = [([0, 1, 2], "big")]
    out = _enforce_max_tokens(groups, blocks, max_tokens=800)
    assert len(out) == 3 or len(out) == 2
    assert all(sum(blocks[i].tokens for i in g[0]) <= 800 or len(g[0]) == 1 for g in out)


def test_single_giant_block_never_splits():
    blocks = [_block(kind="table", chars=10000)]  # one 2500-token table
    out = _enforce_max_tokens([([0], "giant table")], blocks, max_tokens=1024)
    assert len(out) == 1 and out[0][0] == [0]


def test_micro_chunk_merges_into_neighbor():
    blocks = [_block(chars=1200), _block(chars=40)]  # second is ~10 tokens
    out = _merge_micro_chunks([([0], "real"), ([1], "tiny")], blocks)
    assert len(out) == 1 and out[0][0] == [0, 1]
