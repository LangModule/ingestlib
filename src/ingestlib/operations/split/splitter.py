"""split() / asplit() — sections + natural chunks for RAG ingestion.

Three passes: vocabulary discovery (1 call) → per-page labels (parallel) →
within-section chunk boundaries (parallel, skipped for small sections).
Independent of parse: accepts a ParseResult or a raw file path.
"""
import asyncio
import time
from pathlib import Path

from ingestlib.operations.parse.models import ParseResult
from ingestlib.operations.split.models import Chunk, Section, SplitResult, VocabEntry
from ingestlib.operations.split.pages import Block, SplitPage, extract_split_pages
from ingestlib.operations.split.sections import group_pages, label_pages, propose_vocabulary
from ingestlib.operations.split.segmenter import segment_section
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_NOVA_CONCURRENCY = 8

DEFAULT_MAX_CHUNK_TOKENS = 1024


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
) -> SplitResult:
    """Split a document into sections and natural chunks (async).

    source           — a ParseResult from parse(), or a PDF/DOCX/PPTX path (no OCR run)
    category         — optional document-type label (e.g. from classify()) used in
                       each chunk's embedding_text breadcrumb
    max_chunk_tokens — ceiling on chunk size; natural boundaries rule below it
    """
    start = time.perf_counter()
    pages = extract_split_pages(source)
    if not pages:
        return SplitResult(sections=[], vocabulary=[], pages_used=0)

    semaphore = asyncio.Semaphore(_NOVA_CONCURRENCY)

    vocabulary = await propose_vocabulary(pages)
    labels = await label_pages(pages, vocabulary, semaphore)
    grouped = group_pages(pages, labels)
    logger.info(
        "split: %d page(s) → %d section(s): %s",
        len(pages), len(grouped), [name for name, _ in grouped],
    )

    descriptions = {s.name: s.description for s in vocabulary}
    built = list(await asyncio.gather(*[
        _build_section(
            name, descriptions.get(name, ""), section_pages,
            category, max_chunk_tokens, semaphore,
        )
        for name, section_pages in grouped
    ]))
    sections = _renumber_chunks(built)

    result = SplitResult(
        sections=sections,
        vocabulary=[VocabEntry(name=s.name, description=s.description) for s in vocabulary],
        pages_used=len(pages),
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
) -> SplitResult:
    """Split a document into sections and natural chunks.

    Sync wrapper — use asplit() inside an event loop.
    """
    return asyncio.run(
        asplit(source, category=category, max_chunk_tokens=max_chunk_tokens)
    )
