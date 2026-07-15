"""classify() / aclassify() — document-type classification via Nova structured output.

Independent of parse: accepts a ParseResult (enriched markdown + figure crops)
or a raw file path (native text + embedded images, no OCR, no rendering).
≤20 pages classify in one Nova call; larger documents map-reduce over 20-page
chunks (100-page cap).
"""
import asyncio
import time
from pathlib import Path

from pydantic import BaseModel, Field

from ingestlib.foundations.llm import Image as NovaImage
from ingestlib.foundations.llm import achat_structured
from ingestlib.operations.classify.chunker import (
    CHUNK_PAGES,
    PageContent,
    cap_and_chunk,
    extract_pages,
)
from ingestlib.operations.classify.models import CategoryScore, ClassifyResult
from ingestlib.operations.parse.models import ParseResult
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Document images sent alongside text — real embedded images / figure crops,
# never full page renders (text already covers the words; images cover visuals).
_SINGLE_CALL_IMAGES = 4
_COMBINE_CALL_IMAGES = 2

_NOVA_CONCURRENCY = 8

_SYSTEM_PROMPT = (
    "You are a document classification engine. The category names the document "
    "TYPE — what kind of document it is and how it functions (e.g. invoice, "
    "research_paper, insurance_certificate) — never its subject matter. "
    "Labels are lowercase snake_case."
)


class _Verdict(BaseModel):
    """Open-ended classification verdict."""
    category: str = Field(description="snake_case document-type label")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="one or two sentences of justification")


class _Alternative(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class _ConstrainedVerdict(BaseModel):
    """Verdict constrained to caller-supplied categories."""
    category: str = Field(description="one of the allowed labels, or 'uncategorized'")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="one or two sentences of justification")
    alternatives: list[_Alternative] = Field(
        default_factory=list,
        description="other plausible allowed labels ranked by score, best first",
    )


def _format_pages(pages: list[PageContent], start_num: int = 1) -> str:
    placeholder = "(no extractable text — see attached images if any)"
    blocks = [
        f"--- page {i} ---\n{p.text.strip() or placeholder}"
        for i, p in enumerate(pages, start=start_num)
    ]
    return "\n\n".join(blocks)


def _page_images(pages: list[PageContent], limit: int) -> list[NovaImage]:
    images = [NovaImage(data, "png") for p in pages for data in p.images]
    return images[:limit]


def _categories_block(categories: dict[str, str], *, alternatives: bool = True) -> str:
    """Constraint text. alternatives=False for calls using the _Verdict schema,
    which has no alternatives field — instructing the model to fill one would
    be unsatisfiable and degrades structured-output compliance."""
    lines = [f"- {label}: {desc}" for label, desc in categories.items()]
    block = (
        "Allowed categories:\n" + "\n".join(lines) + "\n\n"
        "Pick exactly one label from the list. If none fits, use 'uncategorized'."
    )
    if alternatives:
        block += " Also rank the other plausible labels in 'alternatives' by score."
    return block


def _validate_categories(categories: dict[str, str] | None) -> None:
    if categories is not None and not categories:
        raise ValueError("categories must be a non-empty dict or None")


async def _classify_single(
    pages: list[PageContent],
    categories: dict[str, str] | None,
) -> _Verdict | _ConstrainedVerdict:
    """One Nova call over the whole (capped) document."""
    body = _format_pages(pages)
    images = _page_images(pages, _SINGLE_CALL_IMAGES)
    if categories is None:
        prompt = f"Classify this document.\n\n{body}"
        return await achat_structured(prompt, _Verdict, images=images, system=_SYSTEM_PROMPT)
    prompt = f"Classify this document.\n\n{_categories_block(categories)}\n\n{body}"
    return await achat_structured(
        prompt, _ConstrainedVerdict, images=images, system=_SYSTEM_PROMPT
    )


async def _classify_chunk(
    chunk: list[PageContent],
    chunk_idx: int,
    start_page: int,
    categories: dict[str, str] | None,
    semaphore: asyncio.Semaphore,
) -> _Verdict:
    """Map phase: text-only verdict for one 20-page chunk."""
    constraint = (
        f"\n\n{_categories_block(categories, alternatives=False)}" if categories else ""
    )
    prompt = (
        f"This is a consecutive excerpt (pages {start_page}+) of a larger document. "
        f"Classify the DOCUMENT it comes from.{constraint}\n\n{_format_pages(chunk, start_page)}"
    )
    async with semaphore:
        verdict = await achat_structured(prompt, _Verdict, system=_SYSTEM_PROMPT)
    logger.info("chunk %d verdict: %s (%.2f)", chunk_idx, verdict.category, verdict.confidence)
    return verdict


async def _combine(
    chunk_verdicts: list[_Verdict],
    first_pages: list[PageContent],
    categories: dict[str, str] | None,
) -> _Verdict | _ConstrainedVerdict:
    """Reduce phase: weigh per-chunk verdicts into one final verdict."""
    votes = "\n".join(
        f"- chunk {i}: {v.category} (confidence {v.confidence:.2f}) — {v.reasoning}"
        for i, v in enumerate(chunk_verdicts, start=1)
    )
    constraint = f"\n\n{_categories_block(categories)}" if categories else ""
    images = _page_images(first_pages, _COMBINE_CALL_IMAGES)
    image_note = (
        "Representative images from the document are attached. " if images else ""
    )
    prompt = (
        "Independent classifiers each read a consecutive chunk of one document "
        "and voted:\n\n"
        f"{votes}\n\n"
        f"{image_note}Produce the final classification of the whole document.{constraint}"
    )
    schema = _ConstrainedVerdict if categories else _Verdict
    return await achat_structured(prompt, schema, images=images, system=_SYSTEM_PROMPT)


def _to_result(
    verdict: _Verdict | _ConstrainedVerdict,
    categories: dict[str, str] | None,
    pages_used: int,
) -> ClassifyResult:
    alternatives: list[CategoryScore] = []
    if isinstance(verdict, _ConstrainedVerdict) and categories:
        allowed = set(categories)
        alternatives = [
            CategoryScore(label=a.label, score=a.score)
            for a in verdict.alternatives
            if a.label in allowed and a.label != verdict.category
        ]
        if verdict.category not in allowed and verdict.category != "uncategorized":
            logger.warning(
                "model invented category %r — coercing to 'uncategorized'", verdict.category
            )
            verdict = verdict.model_copy(update={"category": "uncategorized"})
    return ClassifyResult(
        category=verdict.category,
        confidence=verdict.confidence,
        reasoning=verdict.reasoning,
        alternatives=alternatives,
        pages_used=pages_used,
    )


async def aclassify(
    source: ParseResult | Path | str,
    categories: dict[str, str] | None = None,
) -> ClassifyResult:
    """Classify a document's type (async).

    source     — a ParseResult from parse(), or a PDF/DOCX/PPTX path (no OCR run)
    categories — optional {snake_case_label: description}; when given, the result
                 is one of these labels or "uncategorized"
    """
    _validate_categories(categories)
    start = time.perf_counter()
    # loading is blocking work (pypdfium2, LibreOffice subprocess) — keep it
    # off the event loop
    pages = await asyncio.to_thread(extract_pages, source)
    chunks, pages_used = cap_and_chunk(pages)
    logger.info(
        "classify start: %d page(s) used, %d chunk(s), categories=%s",
        pages_used, len(chunks), sorted(categories) if categories else "open-ended",
    )

    if len(chunks) == 1:
        verdict = await _classify_single(chunks[0], categories)
    else:
        semaphore = asyncio.Semaphore(_NOVA_CONCURRENCY)
        tasks = [
            asyncio.ensure_future(
                _classify_chunk(c, i, 1 + i * CHUNK_PAGES, categories, semaphore)
            )
            for i, c in enumerate(chunks)
        ]
        try:
            chunk_verdicts = list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:  # don't leave sibling chunk calls running
                task.cancel()
            raise
        verdict = await _combine(chunk_verdicts, chunks[0], categories)

    result = _to_result(verdict, categories, pages_used)
    logger.info(
        "classify done: %s (%.2f) in %.1fs",
        result.category, result.confidence, time.perf_counter() - start,
    )
    return result


def classify(
    source: ParseResult | Path | str,
    categories: dict[str, str] | None = None,
) -> ClassifyResult:
    """Classify a document's type. Sync wrapper — use aclassify() inside an event loop."""
    return asyncio.run(aclassify(source, categories))
