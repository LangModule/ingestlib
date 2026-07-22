"""Section discovery — Pass 1 (vocabulary) + Pass 2 (per-page labels) + grouping.

Pass 1 reads the whole document once and proposes a FIXED vocabulary of section
categories. Pass 2 classifies every page against that vocabulary in parallel.
Because the vocabulary is fixed, grouping consecutive same-label pages into
sections is pure Python — no synonym merging, no third LLM pass.
"""
import asyncio

from pydantic import BaseModel, Field

from ingestlib.foundations.llm import achat_structured
from ingestlib.operations.split.pages import SplitPage
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Per-page text budget for the vocabulary pass (identity, not fidelity).
_VOCAB_PAGE_LIMIT = 1200
# Per-page text budget for page labeling.
_LABEL_PAGE_LIMIT = 4000

_SYSTEM_PROMPT = (
    "You segment documents into sections. Section names are lowercase snake_case "
    "and describe the ROLE a stretch of pages plays in the document (e.g. "
    "introduction, methods, financial_highlights, appendix). Never name a "
    "section after its layout or format — labels like 'tables', 'figures', "
    "'images', 'text' are forbidden; say what the content DOES in the document."
)


class _VocabSection(BaseModel):
    name: str = Field(description="snake_case section label")
    description: str = Field(description="one sentence: what belongs in this section")


class _Vocabulary(BaseModel):
    sections: list[_VocabSection] = Field(
        description="2-15 section categories that partition this document, in document order"
    )


class _PageLabel(BaseModel):
    category: str = Field(description="exactly one label from the allowed list")


async def propose_vocabulary(pages: list[SplitPage]) -> list[_VocabSection]:
    """Pass 1 — one LLM call over the whole (capped) document."""
    body = "\n\n".join(
        f"--- page {p.page_num} ---\n{p.text.strip()[:_VOCAB_PAGE_LIMIT]}" for p in pages
    )
    prompt = (
        "Read this document and propose the section categories that partition its "
        "pages. Every page must belong to one category. Prefer fewer, broader "
        f"categories for short documents.\n\n{body}"
    )
    vocab = await achat_structured(prompt, _Vocabulary, system=_SYSTEM_PROMPT)
    if not vocab.sections:
        logger.warning("vocabulary pass returned no sections — using single 'document' section")
        return [_VocabSection(name="document", description="entire document")]
    logger.info("vocabulary: %s", [s.name for s in vocab.sections])
    return vocab.sections


async def label_page(
    page: SplitPage,
    vocabulary: list[_VocabSection],
    semaphore: asyncio.Semaphore,
) -> str:
    """Pass 2 — assign one vocabulary label to a page."""
    vocab_block = "\n".join(f"- {s.name}: {s.description}" for s in vocabulary)
    prompt = (
        f"Allowed section labels:\n{vocab_block}\n\n"
        f"Assign page {page.page_num} to exactly one label.\n\n"
        f"--- page {page.page_num} ---\n{page.text.strip()[:_LABEL_PAGE_LIMIT]}"
    )
    async with semaphore:
        label = await achat_structured(prompt, _PageLabel, system=_SYSTEM_PROMPT)
    return label.category


async def label_pages(
    pages: list[SplitPage],
    vocabulary: list[_VocabSection],
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Pass 2 across all pages in parallel, then repair labels outside the vocabulary.

    An invalid label inherits its left neighbor's label (section continuity) —
    deterministic, and logged so drift is visible.
    """
    tasks = [
        asyncio.ensure_future(label_page(p, vocabulary, semaphore)) for p in pages
    ]
    try:
        labels = list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:  # don't leave sibling label calls running
            task.cancel()
        raise
    allowed = {s.name for s in vocabulary}
    for i, label in enumerate(labels):
        if label not in allowed:
            # labels[i-1] is already repaired (left-to-right), so it's always allowed
            fallback = labels[i - 1] if i > 0 else vocabulary[0].name
            logger.warning(
                "page %d got label %r outside vocabulary — using %r",
                pages[i].page_num, label, fallback,
            )
            labels[i] = fallback
    return labels


def group_pages(
    pages: list[SplitPage],
    labels: list[str],
) -> list[tuple[str, list[SplitPage]]]:
    """Consecutive same-label pages → one section. Pure Python, deterministic."""
    groups: list[tuple[str, list[SplitPage]]] = []
    for page, label in zip(pages, labels):
        if groups and groups[-1][0] == label:
            groups[-1][1].append(page)
        else:
            groups.append((label, [page]))
    return groups
