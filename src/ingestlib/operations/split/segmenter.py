"""Pass 3 — natural chunk boundaries within each section.

The LLM proposes topical groupings over a section's blocks; deterministic code
enforces the guarantees the LLM can't be trusted with:

  - chunks are built from whole blocks → tables/figures never split (atomic
    by construction; captions were already folded into their visual)
  - a heading never ends a chunk — it binds to the content below it (the
    ceiling walk preserves this too, budget permitting)
  - chunks over max_tokens get ONE more LLM call proposing budget-aware
    sub-boundaries (so the cut lands where the topic pauses, and each
    sub-chunk gets its own heading); the greedy block-boundary walk still
    runs last, so the ceiling stays a hard guarantee even if the model ignores
    the budget. A single oversized block, e.g. a giant table, stays whole.
  - micro-chunks merge into their neighbor (before the ceiling pass — a hard
    budget cut can still leave a small tail chunk)

Small sections skip the LLM call entirely — they are one natural chunk.
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

# Oversized groups with fewer blocks than this skip the sub-split call —
# with one or two blocks there is at most one possible cut, nothing to choose.
_MIN_SUBSPLIT_BLOCKS = 3

_SYSTEM_PROMPT = (
    "You find natural chunk boundaries in a document section for retrieval. "
    "A chunk is a run of consecutive blocks about ONE thing — a subsection, a "
    "table with its discussion, a complete argument. Never separate content "
    "that must be read together."
)

_SUBSPLIT_SYSTEM_PROMPT = (
    "You split an oversized retrieval chunk at its most natural internal "
    "boundaries. Every sub-chunk must fit a hard token budget AND read as one "
    "coherent unit — never separate a claim from its evidence, a table from "
    "the text that discusses it, or a heading from its content."
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
    """Validate/repair the model's spans into a full partition of [0, n_blocks).

    Overlaps and gaps are repaired by walking spans in order; anything left
    uncovered lands in a final unnamed group. Always returns a valid partition.
    """
    groups: list[tuple[list[int], str]] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start_block):
        # start always resumes at the cursor: overlaps clip forward, gaps fold in
        start, end = cursor, min(span.end_block, n_blocks - 1)
        if end < start:
            continue
        groups.append((list(range(start, end + 1)), span.heading.strip()))
        cursor = end + 1
    if cursor < n_blocks:  # tail the model didn't cover
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
    """Split oversized chunks at block boundaries. A single huge block stays whole.

    A cut must not strand trailing headings — they bind to the content below,
    so they move into the new chunk (when they fit within its budget).
    """
    out: list[tuple[list[int], str]] = []
    for indexes, heading in groups:
        current: list[int] = []
        current_tokens = 0
        for idx in indexes:
            block_tokens = blocks[idx].tokens
            if current and current_tokens + block_tokens > max_tokens:
                carried: list[int] = []
                carried_tokens = 0
                while (
                    current
                    and blocks[current[-1]].kind == "heading"
                    and carried_tokens + blocks[current[-1]].tokens + block_tokens
                    <= max_tokens
                ):
                    moved = current.pop()
                    carried.insert(0, moved)
                    carried_tokens += blocks[moved].tokens
                if current:
                    out.append((current, heading))
                current, current_tokens = carried, carried_tokens
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


async def _subsplit_group(
    section_name: str,
    heading: str,
    indexes: list[int],
    blocks: list[Block],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> list[tuple[list[int], str]]:
    """One LLM call proposing budget-aware sub-boundaries for an oversized group.

    The prompt carries each block's token count so the model can plan cuts that
    fit the budget; the answer goes through the same repair machinery as the
    main pass. Returned groups use DOCUMENT block indexes; callers still run
    _enforce_max_tokens afterwards as the hard guarantee.
    """
    local = [blocks[i] for i in indexes]
    total = sum(b.tokens for b in local)
    min_chunks = -(-total // max_tokens)  # ceil — the fewest sub-chunks that can fit
    listing = "\n".join(
        f"[{j}] ({b.kind}, ~{b.tokens} tokens) {_preview(b)}" for j, b in enumerate(local)
    )
    topic = f" about '{heading}'" if heading else ""
    prompt = (
        f"A retrieval chunk from section '{section_name}'{topic} totals "
        f"~{total} tokens — over the hard budget of {max_tokens} tokens per "
        f"chunk. Split it into at least {min_chunks} sub-chunks, each at most "
        f"{max_tokens} tokens (sum the per-block token counts).\n\n"
        f"Blocks (index, kind, ~tokens, preview):\n\n{listing}\n\n"
        "Group consecutive blocks into coherent sub-chunks within the budget. "
        "Cover every block exactly once, cut where the topic naturally pauses, "
        "and give each sub-chunk a short topic heading."
    )
    async with semaphore:
        seg = await achat_structured(prompt, _Segmentation, system=_SUBSPLIT_SYSTEM_PROMPT)

    groups = _spans_to_groups(seg.chunks, len(local))
    groups = _enforce_heading_binding(groups, local)
    groups = _merge_micro_chunks(groups, local)
    logger.info(
        "sub-split: section %r group of ~%d tokens → %d sub-chunk(s)",
        section_name, total, len(groups),
    )
    # local → document indexes; a sub-chunk the model left unnamed keeps the parent heading
    return [([indexes[j] for j in g], h or heading) for g, h in groups]


async def _enforce_budget(
    section_name: str,
    groups: list[tuple[list[int], str]],
    blocks: list[Block],
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> list[tuple[list[int], str]]:
    """Two-tier ceiling: the LLM places the cuts, the greedy walk guarantees them.

    Groups within budget pass through untouched (no LLM call). Oversized
    groups with enough blocks to give the model a real choice get one
    sub-split call; on any failure the group falls through unchanged. The
    final _enforce_max_tokens pass is always the last word.
    """

    async def resolve(indexes: list[int], heading: str) -> list[tuple[list[int], str]]:
        total = sum(blocks[i].tokens for i in indexes)
        if total <= max_tokens or len(indexes) < _MIN_SUBSPLIT_BLOCKS:
            return [(indexes, heading)]
        try:
            return await _subsplit_group(
                section_name, heading, indexes, blocks, max_tokens, semaphore
            )
        except Exception:
            logger.warning(
                "sub-split call failed for section %r — falling back to "
                "block-boundary cut", section_name, exc_info=True,
            )
            return [(indexes, heading)]

    resolved = await asyncio.gather(*[resolve(idxs, h) for idxs, h in groups])
    flat = [group for sub in resolved for group in sub]
    return _enforce_max_tokens(flat, blocks, max_tokens)


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
        return await _enforce_budget(
            section_name, [(list(range(len(blocks))), "")], blocks, max_tokens, semaphore
        )

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
    # merge BEFORE the ceiling passes — merging afterwards could push a chunk
    # back over max_tokens; the ceiling must be the last word.
    groups = _merge_micro_chunks(groups, blocks)
    groups = await _enforce_budget(section_name, groups, blocks, max_tokens, semaphore)
    logger.info(
        "section %r: %d blocks → %d chunk(s)", section_name, len(blocks), len(groups)
    )
    return groups
