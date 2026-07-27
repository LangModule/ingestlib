"""Source preparation for extract — region-tagged pages and call windows.

The ParseResult path tags every block with its full citation marker
(`[p4:r2] …`) — the model cites by copying markers verbatim, and the
citations resolve to real bounding boxes. The raw-path path reuses
classify's no-OCR content loading — page-level provenance only, and the
same "no text layer → parse() first" guard for scans.
"""
from pathlib import Path
from typing import NamedTuple

from ingestlib.operations.classify.chunker import (
    extract_pages as _load_native_pages,
)
from ingestlib.operations.classify.chunker import (
    select_pages,
)
from ingestlib.operations.parse.models import ParseResult
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Extraction never reads past this many pages — the same posture as classify.
MAX_PAGES = 100

# mode="many": instances (receipts, line items) live locally — small windows
# with one page of overlap so an instance split across a boundary is seen
# whole by at least one window.
MANY_WINDOW_PAGES = 4
MANY_WINDOW_OVERLAP = 1

# mode="one": one call covers this much; longer documents run per-window and
# merge field-by-field.
ONE_WINDOW_PAGES = 20

# Extraction needs detail (line items, totals in footers) — a bigger per-page
# budget than classification's skim.
PAGE_TEXT_LIMIT = 12000


class SourcePage(NamedTuple):
    """One page as the extractor sees it.

    region_ids is None on the native path — citations are then page-level.
    region_texts maps region id → that block's text, for grounding checks.
    """

    page_num: int
    text: str
    region_ids: frozenset[int] | None
    region_texts: dict[int, str]
    images: list[bytes]


def build_pages(source: ParseResult | Path | str) -> list[SourcePage]:
    """Normalize either input into region-tagged (or plain) page records."""
    if isinstance(source, ParseResult):
        pages: list[SourcePage] = []
        for p in source.pages:
            lines: list[str] = []
            texts: dict[int, str] = {}
            for r in p.regions:
                body = (r.content or r.text).strip()
                if not body:
                    continue
                # the marker IS the citation format — the model just copies it
                lines.append(f"[p{p.page_num}:r{r.region_id}] {body}")
                texts[r.region_id] = body
            pages.append(SourcePage(
                page_num=p.page_num,
                text="\n".join(lines)[:PAGE_TEXT_LIMIT],
                region_ids=frozenset(texts),
                region_texts=texts,
                images=[f.image_bytes for f in p.figures],
            ))
        return pages

    return [
        SourcePage(
            page_num=cp.page_num,
            text=cp.text,
            region_ids=None,
            region_texts={},
            images=cp.images,
        )
        for cp in _load_native_pages(source)
    ]


def prepare_windows(
    pages: list[SourcePage],
    mode: str,
    target_pages: str | None,
) -> tuple[list[list[SourcePage]], int]:
    """Select, cap, and window the pages. Returns (windows, pages_used)."""
    pages = select_pages(pages, target_pages, None)
    if len(pages) > MAX_PAGES:
        logger.warning(
            "document has %d pages — extracting from the first %d only",
            len(pages), MAX_PAGES,
        )
        pages = pages[:MAX_PAGES]

    if mode == "many":
        step = MANY_WINDOW_PAGES - MANY_WINDOW_OVERLAP
        windows = [
            pages[i : i + MANY_WINDOW_PAGES] for i in range(0, len(pages), step)
        ]
        # the stride can leave a trailing window fully contained in the
        # previous one — drop it
        windows = [
            w for i, w in enumerate(windows)
            if i == 0 or w[-1].page_num > windows[i - 1][-1].page_num
        ]
    else:
        windows = [
            pages[i : i + ONE_WINDOW_PAGES]
            for i in range(0, len(pages), ONE_WINDOW_PAGES)
        ]
    return windows, len(pages)


def format_window(pages: list[SourcePage]) -> str:
    """Prompt body for one window — pages delimited, blocks region-tagged."""
    placeholder = "(no extractable text on this page)"
    blocks = [
        f"--- page {p.page_num} ---\n{p.text.strip() or placeholder}" for p in pages
    ]
    return "\n\n".join(blocks)
