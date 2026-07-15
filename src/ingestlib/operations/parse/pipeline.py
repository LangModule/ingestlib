"""parse() / aparse() orchestrator — one mode, the best output we can produce.

Per page: PaddleOCR-VL (layout + recognition) → Nova enrichment of charts/figures
→ Nova review (per-region corrections) → markdown assembly.

The OCR stage is GPU-bound and runs one page at a time; Nova stages run
concurrently behind it, so cloud latency adds almost nothing to wall-clock.
"""
import asyncio
import time
from pathlib import Path
from typing import Any

from ingestlib.foundations.ocr import paddle_vl
from ingestlib.operations.parse.assembler import assemble_markdown, assemble_text
from ingestlib.operations.parse.detector import detect_format
from ingestlib.operations.parse.enricher import enrich_page
from ingestlib.operations.parse.loaders import (
    LoadedPage,
    load_office,
    load_pdf,
)
from ingestlib.operations.parse.models import PageResult, ParseResult, SourceFormat
from ingestlib.operations.parse.reviewer import review_page
from ingestlib.utils.files import sha256_of_file
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# The local OCR pipeline serializes on the GPU — one page in flight keeps it
# saturated without stacking blocked threads.
_OCR_CONCURRENCY = 1

# Concurrent Nova calls across all pages (enrichment + review share the pool).
_NOVA_CONCURRENCY = 8


def _load(
    path: Path,
    fmt: SourceFormat,
    dpi: int,
) -> tuple[list[LoadedPage], dict[str, Any], bool]:
    """Format-specific loading. Returns (pages, metadata, was_converted)."""
    if fmt == "pdf":
        pages, meta = load_pdf(path, render=True, dpi=dpi)
        return pages, meta, False
    # docx / pptx via LibreOffice → PDF
    pages, meta = load_office(path, render=True, dpi=dpi)
    return pages, meta, True


async def _process_page(
    lp: LoadedPage,
    page_num: int,
    dpi: int,
    ocr_semaphore: asyncio.Semaphore,
    nova_semaphore: asyncio.Semaphore,
) -> PageResult:
    """One page through the full pipeline: OCR → enrich → review → assemble."""
    if lp.image_bytes is None:
        raise RuntimeError(f"page {page_num} has no rendered image")

    async with ocr_semaphore:
        t0 = time.perf_counter()
        layout = await paddle_vl.arun_full_pipeline(lp.image_bytes)
        logger.info(
            "page %d: OCR done in %.1fs (%d regions)",
            page_num, time.perf_counter() - t0, len(layout.regions),
        )

    if not layout.regions:
        return PageResult(
            page_num=page_num,
            native_text=lp.native_text,
            image_bytes=lp.image_bytes,
            image_dpi=dpi,
            page_width=layout.page_width,
            page_height=layout.page_height,
        )

    regions, figures = await enrich_page(
        layout.regions, lp.image_bytes, layout.page_height, nova_semaphore
    )
    regions = await review_page(regions, lp.image_bytes, lp.native_text, nova_semaphore)

    return PageResult(
        page_num=page_num,
        text=assemble_text(regions),
        markdown=assemble_markdown(regions, figures, page_num),
        regions=regions,
        figures=figures,
        native_text=lp.native_text,
        image_bytes=lp.image_bytes,
        image_dpi=dpi,
        page_width=layout.page_width,
        page_height=layout.page_height,
    )


async def aparse(path: Path | str, *, dpi: int = 200) -> ParseResult:
    """Parse a document into a ParseResult (async).

    path — PDF/DOCX/PPTX to parse
    dpi  — page render resolution; 200 balances OCR accuracy against
           VLM token cost and memory
    """
    start = time.perf_counter()
    path = Path(path)
    fmt = detect_format(path)
    checksum = sha256_of_file(path)
    logger.info("parse start: path=%s format=%s", path, fmt)

    loaded_pages, metadata, was_converted = await asyncio.to_thread(_load, path, fmt, dpi)
    logger.info(
        "loaded %d page(s) via %s loader (was_converted=%s)",
        len(loaded_pages), fmt, was_converted,
    )

    ocr_semaphore = asyncio.Semaphore(_OCR_CONCURRENCY)
    nova_semaphore = asyncio.Semaphore(_NOVA_CONCURRENCY)
    tasks = [
        asyncio.ensure_future(_process_page(
            lp, page_num=i, dpi=dpi,
            ocr_semaphore=ocr_semaphore, nova_semaphore=nova_semaphore,
        ))
        for i, lp in enumerate(loaded_pages, start=1)
    ]
    try:
        pages = list(await asyncio.gather(*tasks))
    except BaseException:
        # one failed page must not leave siblings running as orphans, holding
        # the OCR semaphore and spending Nova calls
        for task in tasks:
            task.cancel()
        raise

    duration = time.perf_counter() - start
    logger.info("parse complete: pages=%d duration=%.2fs", len(pages), duration)
    return ParseResult(
        pages=pages,
        source_path=path,
        source_format=fmt,
        was_converted=was_converted,
        source_metadata=metadata,
        source_checksum=checksum,
        parse_duration_seconds=duration,
    )


def parse(path: Path | str, *, dpi: int = 200) -> ParseResult:
    """Parse a document into a ParseResult.

    Synchronous wrapper around aparse(). If you're already inside an event
    loop, use aparse() instead.
    """
    return asyncio.run(aparse(path, dpi=dpi))
