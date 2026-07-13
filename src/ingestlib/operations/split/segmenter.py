"""Pass 3 — natural chunk boundaries within each section.

Nova proposes topical groupings over a section's blocks; deterministic code
enforces the guarantees the LLM can't be trusted with:

  - chunks are built from whole blocks → tables/figures never split (atomic
    by construction; captions were already folded into their visual)
  - a heading never ends a chunk — it binds to the content below it
  - chunks over max_tokens split at block boundaries (a single oversized
    block, e.g. a giant table, stays whole)
  - micro-chunks merge into their neighbor

Small sections skip the Nova call entirely — they are one natural chunk.
"""
import asyncio

from pydantic import BaseModel, Field

from ingestlib.foundations.llm import achat_structured
from ingestlib.operations.split.pages import Block
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Sections at or below this token size are a single chunk — no boundary call.
_SINGLE_CHUNK_TOKENS = 700

# Chunks smaller than this merge into a neighbor.
_MIN_CHUNK_TOKENS = 50

# Per-block preview length in the boundary prompt.
_BLOCK_PREVIEW_CHARS = 240

_SYSTEM_PROMPT = (
    "You find natural chunk boundaries in a document section for retrieval. "
    "A chunk is a run of consecutive blocks about ONE thing — a subsection, a "
    "table with its discussion, a complete argument. Never separate content "
    "that must be read together."
)


class _ChunkSpan(BaseModel):
    start_block: int = Field(description="index of the first block in this chunk")
    end_block: int = Field(description="index of the last block in this chunk (inclusive)")
    heading: str = Field(description="3-8 word topic label for this chunk")


class _Segmentation(BaseModel):
    chunks: list[_ChunkSpan] = Field(
        description="contiguous spans covering every block exactly once, in order"
    )


def _preview(block: Block) -> str:
    text = " ".join(block.markdown.split())
    return text[:_BLOCK_PREVIEW_CHARS]


def _spans_to_groups(
    spans: list[_ChunkSpan], n_blocks: int
) -> list[tuple[list[int], str]]:
    """Validate/repair Nova's spans into a full partition of [0, n_blocks).

    Overlaps and gaps are repaired by walking spans in order; anything left
    uncovered lands in a final unnamed group. Always returns a valid partition.
    """
    groups: list[tuple[list[int], str]] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start_block):
        start, end = max(span.start_block, cursor), min(span.end_block, n_blocks - 1)
        if start > cursor:  # gap — fold skipped blocks into this chunk
            start = cursor
        if end < start:
            continue
        groups.append((list(range(start, end + 1)), span.heading.strip()))
        cursor = end + 1
    if cursor < n_blocks:  # tail Nova didn't cover
        groups.append((list(range(cursor, n_blocks)), ""))
    return groups


def _enforce_heading_binding(
    groups: list[tuple[list[int], str]], blocks: list[Block]
) -> list[tuple[list[int], str]]:
    """A chunk must not end with a heading block — move it to the next chunk."""
    for i in range(len(groups) - 1):
        indexes, heading = groups[i]
        while indexes and blocks[indexes[-1]].kind == "heading":
            moved = indexes.pop()
            groups[i + 1][0].insert(0, moved)
        groups[i] = (indexes, heading)
    return [g for g in groups if g[0]]


def _enforce_max_tokens(
    groups: list[tuple[list[int], str]], blocks: list[Block], max_tokens: int
) -> list[tuple[list[int], str]]:
    """Split oversized chunks at block boundaries. A single huge block stays whole."""
    out: list[tuple[list[int], str]] = []
    for indexes, heading in groups:
        current: list[int] = []
        current_tokens = 0
        for idx in indexes:
            block_tokens = blocks[idx].tokens
            if current and current_tokens + block_tokens > max_tokens:
                out.append((current, heading))
                current, current_tokens = [], 0
            current.append(idx)
            current_tokens += block_tokens
        if current:
            out.append((current, heading))
    return out


def _merge_micro_chunks(
    groups: list[tuple[list[int], str]], blocks: list[Block]
) -> list[tuple[list[int], str]]:
    """Chunks under _MIN_CHUNK_TOKENS merge into the previous chunk (or next)."""
    out: list[tuple[list[int], str]] = []
    for indexes, heading in groups:
        tokens = sum(blocks[i].tokens for i in indexes)
        if tokens < _MIN_CHUNK_TOKENS and out:
            prev_indexes, prev_heading = out[-1]
            out[-1] = (prev_indexes + indexes, prev_heading)
        else:
            out.append((indexes, heading))
    if len(out) >= 2:
        first_indexes, first_heading = out[0]
        if sum(blocks[i].tokens for i in first_indexes) < _MIN_CHUNK_TOKENS:
            nxt_indexes, nxt_heading = out[1]
            out[0:2] = [(first_indexes + nxt_indexes, nxt_heading or first_heading)]
    return out


async def segment_section(
    section_name: str,
    blocks: list[Block],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> list[tuple[list[int], str]]:
    """Return the section's chunks as (block indexes, heading) groups, in order."""
    if not blocks:
        return []

    total_tokens = sum(b.tokens for b in blocks)
    if total_tokens <= _SINGLE_CHUNK_TOKENS or len(blocks) == 1:
        return _enforce_max_tokens([(list(range(len(blocks))), "")], blocks, max_tokens)

    listing = "\n".join(
        f"[{i}] ({b.kind}) {_preview(b)}" for i, b in enumerate(blocks)
    )
    prompt = (
        f"Section '{section_name}' has {len(blocks)} blocks (index, kind, preview):\n\n"
        f"{listing}\n\n"
        "Group consecutive blocks into natural retrieval chunks. Cover every "
        "block exactly once, keep tables/figures with the text that discusses "
        "them, and give each chunk a short topic heading."
    )
    async with semaphore:
        seg = await achat_structured(prompt, _Segmentation, system=_SYSTEM_PROMPT)

    groups = _spans_to_groups(seg.chunks, len(blocks))
    groups = _enforce_heading_binding(groups, blocks)
    # merge BEFORE the ceiling pass — merging afterwards could push a chunk
    # back over max_tokens; the ceiling must be the last word.
    groups = _merge_micro_chunks(groups, blocks)
    groups = _enforce_max_tokens(groups, blocks, max_tokens)
    logger.info(
        "section %r: %d blocks → %d chunk(s)", section_name, len(blocks), len(groups)
    )
    return groups
