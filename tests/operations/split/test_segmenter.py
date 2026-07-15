"""Boundary-repair and guarantee logic — pure, always run."""
import asyncio

from ingestlib.operations.split.pages import Block
from ingestlib.operations.split.segmenter import (
    _ChunkSpan,
    _enforce_budget,
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


def test_ceiling_cut_carries_trailing_heading_into_next_chunk():
    # text(500) heading(10) text(500): the cut lands after the heading — it
    # must move down with the content it binds to, not end the first chunk.
    blocks = [_block(chars=2000), _block(kind="heading", chars=40), _block(chars=2000)]
    out = _enforce_max_tokens([([0, 1, 2], "big")], blocks, max_tokens=600)
    assert out[0][0] == [0], "first chunk must not end with the heading"
    assert out[1][0] == [1, 2], "heading moved down with its content"


def test_ceiling_cut_leaves_heading_when_carry_would_bust_budget():
    # heading + next block exceed the budget together; the ceiling wins.
    blocks = [_block(chars=2000), _block(kind="heading", chars=400), _block(chars=2400)]
    out = _enforce_max_tokens([([0, 1, 2], "big")], blocks, max_tokens=650)
    covered = [i for g in out for i in g[0]]
    assert covered == [0, 1, 2]
    assert all(
        sum(blocks[i].tokens for i in g[0]) <= 650 or len(g[0]) == 1 for g in out
    )


def test_single_giant_block_never_splits():
    blocks = [_block(kind="table", chars=10000)]  # one 2500-token table
    out = _enforce_max_tokens([([0], "giant table")], blocks, max_tokens=1024)
    assert len(out) == 1 and out[0][0] == [0]


def test_micro_chunk_merges_into_neighbor():
    blocks = [_block(chars=1200), _block(chars=40)]  # second is ~10 tokens
    out = _merge_micro_chunks([([0], "real"), ([1], "tiny")], blocks)
    assert len(out) == 1 and out[0][0] == [0, 1]


def _run_enforce_budget(groups, blocks, max_tokens):
    return asyncio.run(
        _enforce_budget("test_section", groups, blocks, max_tokens, asyncio.Semaphore(1))
    )


def test_budget_within_limit_makes_no_llm_call():
    # Under-budget groups pass through untouched — if this tried to call Nova
    # it would blow up on missing config, which is exactly the point.
    blocks = [_block(chars=800), _block(chars=800)]  # 200 tok each
    out = _run_enforce_budget([([0, 1], "fine")], blocks, max_tokens=768)
    assert out == [([0, 1], "fine")]


def test_budget_two_block_group_skips_llm_and_cuts_greedily():
    # Only one possible cut exists — no semantic choice, no call.
    blocks = [_block(chars=2400), _block(chars=2400)]  # 600 tok each
    out = _run_enforce_budget([([0, 1], "big")], blocks, max_tokens=768)
    assert [g[0] for g in out] == [[0], [1]]
    assert all(h == "big" for _, h in out)


def test_budget_single_giant_block_stays_whole_without_llm():
    blocks = [_block(kind="table", chars=10000)]
    out = _run_enforce_budget([([0], "giant table")], blocks, max_tokens=768)
    assert out == [([0], "giant table")]
