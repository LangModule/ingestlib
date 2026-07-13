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
    """One page as the classifier sees it: text + the page's actual images."""
    text: str
    images: list[bytes]


def extract_pages(source: ParseResult | Path | str) -> list[PageContent]:
    """Normalize either input into per-page (text, images) records."""
    if isinstance(source, ParseResult):
        return [
            PageContent(
                text=(p.markdown or p.text or p.native_text)[:PAGE_TEXT_LIMIT],
                images=[f.image_bytes for f in p.figures],
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
        PageContent(text=cp.text[:PAGE_TEXT_LIMIT], images=cp.images) for cp in loaded
    ]


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
