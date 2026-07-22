"""split() / asplit() — sections + natural chunks for RAG ingestion.

Three passes: vocabulary discovery (1 call — skipped when the caller
supplies a vocabulary) → per-page labels (parallel) → within-section chunk
boundaries (parallel, skipped for small sections). With a user vocabulary,
`unmatched` decides what happens to pages that fit no category:
"other" (default — an honest `other` section), "require" (every page lands
in a category via left-neighbor repair), or "skip" (dropped entirely).
Unset arguments resolve from rules.yaml's `split:` preset. Independent of
parse: accepts a ParseResult or a raw file path.
"""
import asyncio
import time
from pathlib import Path

from ingestlib.config import get_config
from ingestlib.operations.parse.models import ParseResult
from ingestlib.operations.split.models import Chunk, Section, SplitResult, VocabEntry
from ingestlib.operations.split.pages import Block, SplitPage, extract_split_pages
from ingestlib.operations.split.sections import (
    OTHER_LABEL,
    group_pages,
    label_pages,
    make_vocabulary,
    propose_vocabulary,
)
from ingestlib.operations.split.segmenter import segment_section
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_LLM_CONCURRENCY = 8

DEFAULT_MAX_CHUNK_TOKENS = 768

# The split-categories ceiling — past this, per-label discrimination degrades.
MAX_CATEGORIES = 50

_UNMATCHED_MODES = ("require", "other", "skip")

_OTHER_DESCRIPTION = "pages matching no user category"


def _resolve_settings(
    vocabulary: dict[str, str] | None,
    unmatched: str | None,
) -> tuple[dict[str, str] | None, str]:
    """Fill unset arguments from rules.yaml's `split:` preset.

    None means "use the preset"; an explicit {} forces LLM discovery even
    when a preset exists. Explicit arguments always win over the preset."""
    preset = get_config().split
    if vocabulary is None:
        vocabulary = dict(preset.categories) or None
    elif not vocabulary:
        vocabulary = None
    if vocabulary is not None and len(vocabulary) > MAX_CATEGORIES:
        raise ValueError(
            f"{len(vocabulary)} split categories — the limit is {MAX_CATEGORIES}"
        )
    if unmatched is None:
        unmatched = preset.unmatched or "other"
    if unmatched not in _UNMATCHED_MODES:
        raise ValueError(
            f"unmatched must be one of {list(_UNMATCHED_MODES)}, got {unmatched!r}"
        )
    if vocabulary is None and unmatched != "other":
        raise ValueError(
            f"unmatched={unmatched!r} applies only with a user vocabulary — a "
            f"discovered vocabulary covers every page by construction"
        )
    return vocabulary, unmatched


def _dominant_kind(blocks: list[Block]) -> str:
    kinds = {b.kind for b in blocks if b.kind != "heading"}
    if kinds == {"table"}:
        return "table"
    if kinds == {"figure"}:
        return "figure"
    if len(kinds) > 1:
        return "mixed"
    return "text"


def _breadcrumb(category: str | None, section: str, heading: str) -> str:
    parts = [p for p in (category, section, heading) if p]
    return f"[{' › '.join(parts)}]"


def _build_chunk(
    chunk_id: int,
    section_name: str,
    heading: str,
    blocks: list[Block],
    category: str | None,
) -> Chunk:
    markdown = "\n\n".join(b.markdown for b in blocks)
    text = "\n".join(b.text for b in blocks if b.text)
    region_ids: dict[int, list[int]] = {}
    for b in blocks:
        if b.region_ids:
            region_ids.setdefault(b.page_num, []).extend(b.region_ids)
    return Chunk(
        chunk_id=chunk_id,
        section=section_name,
        heading=heading,
        text=text,
        markdown=markdown,
        embedding_text=f"{_breadcrumb(category, section_name, heading)}\n\n{markdown}",
        pages=sorted({b.page_num for b in blocks}),
        region_ids=region_ids,
        kind=_dominant_kind(blocks),
        token_estimate=sum(b.tokens for b in blocks),
    )


async def _build_section(
    name: str,
    description: str,
    pages: list[SplitPage],
    category: str | None,
    max_chunk_tokens: int,
    semaphore: asyncio.Semaphore,
) -> Section:
    """Segment one section's blocks into chunks and assemble the Section."""
    blocks = [b for p in pages for b in p.blocks]
    groups = await segment_section(name, blocks, max_chunk_tokens, semaphore)
    chunks = [
        _build_chunk(0, name, heading, [blocks[i] for i in indexes], category)
        for indexes, heading in groups
    ]
    return Section(
        name=name,
        description=description,
        pages=[p.page_num for p in pages],
        text="\n".join(p.text for p in pages if p.text),
        markdown="\n\n".join(b.markdown for b in blocks),
        chunks=chunks,
    )


def _renumber_chunks(sections: list[Section]) -> list[Section]:
    """Assign document-wide chunk_ids in reading order (models are frozen — rebuild)."""
    out: list[Section] = []
    next_id = 0
    for s in sections:
        renumbered = []
        for c in s.chunks:
            renumbered.append(c.model_copy(update={"chunk_id": next_id}))
            next_id += 1
        out.append(s.model_copy(update={"chunks": renumbered}))
    return out


async def asplit(
    source: ParseResult | Path | str,
    *,
    category: str | None = None,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    vocabulary: dict[str, str] | None = None,
    unmatched: str | None = None,
) -> SplitResult:
    """Split a document into sections and natural chunks (async).

    source           — a ParseResult from parse(), or a PDF/DOCX/PPTX path (no OCR run)
    category         — optional document-type label (e.g. from classify()) used in
                       each chunk's embedding_text breadcrumb
    max_chunk_tokens — ceiling on chunk size; natural boundaries rule below it
    vocabulary       — optional {section: description}, max 50; when given, Pass 1
                       is skipped and pages label against YOUR sections. None uses
                       rules.yaml's `split:` preset; {} forces LLM discovery.
    unmatched        — pages fitting no user category: "other" (default — an
                       honest `other` section) | "require" (left-neighbor repair)
                       | "skip" (dropped). None uses the preset.
    """
    user_vocabulary, unmatched = _resolve_settings(vocabulary, unmatched)
    start = time.perf_counter()
    pages = extract_split_pages(source)
    if not pages:
        return SplitResult(sections=[], vocabulary=[], pages_used=0)

    semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

    if user_vocabulary is not None:
        vocab = make_vocabulary(user_vocabulary)
        logger.info(
            "split: user vocabulary (%d categories, unmatched=%s) — Pass 1 skipped",
            len(vocab), unmatched,
        )
        labels = await label_pages(pages, vocab, semaphore, unmatched=unmatched)
    else:
        vocab = await propose_vocabulary(pages)
        labels = await label_pages(pages, vocab, semaphore)

    pages_read = len(pages)
    if user_vocabulary is not None and unmatched == "skip":
        kept = [(p, label) for p, label in zip(pages, labels) if label != OTHER_LABEL]
        if len(kept) < len(pages):
            logger.info(
                "split: skipped %d unmatched page(s) of %d", len(pages) - len(kept), len(pages),
            )
        pages = [p for p, _ in kept]
        labels = [label for _, label in kept]
        if not pages:
            return SplitResult(
                sections=[],
                vocabulary=[VocabEntry(name=s.name, description=s.description) for s in vocab],
                pages_used=pages_read,
            )

    grouped = group_pages(pages, labels)
    logger.info(
        "split: %d page(s) → %d section(s): %s",
        len(pages), len(grouped), [name for name, _ in grouped],
    )

    descriptions = {s.name: s.description for s in vocab}
    descriptions.setdefault(OTHER_LABEL, _OTHER_DESCRIPTION)
    tasks = [
        asyncio.ensure_future(_build_section(
            name, descriptions.get(name, ""), section_pages,
            category, max_chunk_tokens, semaphore,
        ))
        for name, section_pages in grouped
    ]
    try:
        built = list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:  # don't leave sibling section builds running
            task.cancel()
        raise
    sections = _renumber_chunks(built)

    result_vocab = [VocabEntry(name=s.name, description=s.description) for s in vocab]
    if any(s.name == OTHER_LABEL for s in sections) and all(
        v.name != OTHER_LABEL for v in result_vocab
    ):
        result_vocab.append(VocabEntry(name=OTHER_LABEL, description=_OTHER_DESCRIPTION))

    result = SplitResult(
        sections=sections,
        vocabulary=result_vocab,
        pages_used=pages_read,
    )
    logger.info(
        "split done: %d section(s), %d chunk(s) in %.1fs",
        len(result.sections), len(result.chunks), time.perf_counter() - start,
    )
    return result


def split(
    source: ParseResult | Path | str,
    *,
    category: str | None = None,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    vocabulary: dict[str, str] | None = None,
    unmatched: str | None = None,
) -> SplitResult:
    """Split a document into sections and natural chunks.

    Sync wrapper — use asplit() inside an event loop.
    """
    return asyncio.run(asplit(
        source, category=category, max_chunk_tokens=max_chunk_tokens,
        vocabulary=vocabulary, unmatched=unmatched,
    ))
