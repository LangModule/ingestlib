"""Page extraction and chunking for classify — works with or without a prior parse.

extract_pages() accepts either a ParseResult (enriched markdown + the figure/chart
crops parse already extracted) or a file path (native text + embedded raster
images pulled straight from the PDF objects — no OCR, no VL model, no page
rendering). Both normalize to the same PageContent shape, so the classifier is
input-agnostic and only ever sees real content: text plus actual pictures.
"""
from pathlib import Path
from typing import NamedTuple

from ingestlib.operations.parse.detector import detect_format
from ingestlib.operations.parse.loaders import load_office_content, load_pdf_content
from ingestlib.operations.parse.models import ParseResult
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Classification never reads past this many pages — the hard cap.
MAX_PAGES = 100

# Map-reduce chunk size: docs over this many pages classify per-chunk, then combine.
CHUNK_PAGES = 20

# Per-page text budget in the prompt. Classification doesn't need a full
# 15k-char timetable; the first stretch of a page identifies it.
PAGE_TEXT_LIMIT = 8000


class PageContent(NamedTuple):
    """One page as the classifier sees it: text + the page's actual images.

    page_num is the ORIGINAL 1-based number, so prompts stay truthful even
    after target_pages selects a sparse subset."""
    text: str
    images: list[bytes]
    page_num: int = 0


def extract_pages(source: ParseResult | Path | str) -> list[PageContent]:
    """Normalize either input into per-page (text, images, page_num) records."""
    if isinstance(source, ParseResult):
        return [
            PageContent(
                text=(p.markdown or p.text or p.native_text)[:PAGE_TEXT_LIMIT],
                images=[f.image_bytes for f in p.figures],
                page_num=p.page_num,
            )
            for p in source.pages
        ]

    path = Path(source)
    fmt = detect_format(path)
    if fmt == "pdf":
        loaded, _ = load_pdf_content(path)
    else:  # docx / pptx
        loaded, _ = load_office_content(path)
    logger.info(
        "classify standalone load: %s (%d pages, %d embedded images, no OCR)",
        path.name, len(loaded), sum(len(p.images) for p in loaded),
    )
    return [
        PageContent(text=cp.text[:PAGE_TEXT_LIMIT], images=cp.images, page_num=i)
        for i, cp in enumerate(loaded, start=1)
    ]


def parse_page_spec(spec: str) -> list[int]:
    """'1, 3, 5-7' → [1, 3, 5, 6, 7]. 1-based, whitespace-tolerant, deduped,
    ascending. Raises ValueError on malformed tokens or an empty spec."""
    pages: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                lo_text, _, hi_text = token.partition("-")
                lo, hi = int(lo_text), int(hi_text)
                if lo < 1 or hi < lo:
                    raise ValueError
                pages.update(range(lo, hi + 1))
            else:
                page = int(token)
                if page < 1:
                    raise ValueError
                pages.add(page)
        except ValueError:
            raise ValueError(
                f"bad target_pages token {token!r} — expected a 1-based page "
                f"number or range like '1,3,5-7'"
            ) from None
    if not pages:
        raise ValueError(f"target_pages {spec!r} selects no pages")
    return sorted(pages)


def select_pages(
    pages: list[PageContent],
    target_pages: str | None,
    max_pages: int | None,
) -> list[PageContent]:
    """Apply the caller's page settings: target_pages picks, max_pages caps.

    Targets beyond the document's length are silently dropped; a spec that
    selects nothing at all raises, because classifying zero pages is a
    caller mistake, not a verdict."""
    if target_pages:
        wanted = parse_page_spec(target_pages)
        selected = [pages[n - 1] for n in wanted if n <= len(pages)]
        if not selected:
            raise ValueError(
                f"target_pages {target_pages!r} selects no pages of a "
                f"{len(pages)}-page document"
            )
        pages = selected
    if max_pages and max_pages > 0:
        pages = pages[:max_pages]
    return pages


def cap_and_chunk(pages: list[PageContent]) -> tuple[list[list[PageContent]], int]:
    """Apply the 100-page cap, then split into chunks of CHUNK_PAGES.

    Returns (chunks, pages_used). One chunk means the single-call path.
    """
    if len(pages) > MAX_PAGES:
        logger.warning(
            "document has %d pages — classifying the first %d only", len(pages), MAX_PAGES
        )
        pages = pages[:MAX_PAGES]
    chunks = [pages[i : i + CHUNK_PAGES] for i in range(0, len(pages), CHUNK_PAGES)]
    return chunks, len(pages)
