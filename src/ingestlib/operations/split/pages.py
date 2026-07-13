"""Page and block extraction for split — works with or without a prior parse.

A Block is the atomic unit of chunking: one region's markdown (ParseResult
input) or one paragraph (standalone input). Chunks are built from whole blocks,
so a table or figure can never be split down the middle by construction.
"""
from pathlib import Path
from typing import NamedTuple

from ingestlib.operations.parse.assembler import render_region
from ingestlib.operations.parse.detector import detect_format
from ingestlib.operations.parse.loaders import load_office_content, load_pdf_content
from ingestlib.operations.parse.models import ParseResult
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Split never reads past this many pages — the hard cap.
MAX_PAGES = 500

# Region types folded into the visual block they caption.
_CAPTION_TYPES = ("figure_caption", "table_caption")

_KIND_BY_REGION = {
    "title": "heading",
    "table": "table",
    "chart": "figure",
    "figure": "figure",
}


class Block(NamedTuple):
    """One atomic content unit.

    region_ids is the provenance of everything merged into this block
    (a figure block carries its caption regions too); empty when standalone.
    """
    page_num: int
    kind: str                # heading | text | table | figure
    markdown: str
    text: str
    region_ids: tuple[int, ...]

    @property
    def tokens(self) -> int:
        return max(1, len(self.markdown) // 4)


class SplitPage(NamedTuple):
    """One page as split sees it: its text (for section labeling) + its blocks."""
    page_num: int
    text: str
    blocks: list[Block]


def _merge_into(host: Block, caption: Block, caption_first: bool) -> Block:
    """Fold a caption block into its visual, preserving reading order."""
    first, second = (caption, host) if caption_first else (host, caption)
    return host._replace(
        markdown=f"{first.markdown}\n\n{second.markdown}",
        text=f"{first.text}\n{second.text}".strip(),
        region_ids=first.region_ids + second.region_ids,
    )


def _fold_captions(blocks: list[Block]) -> list[Block]:
    """Make caption + visual atomic regardless of which comes first on the page.

    A caption folds into the visual directly after it (captions above tables)
    or, failing that, the visual directly before it (captions below figures).
    A caption with no adjacent visual stays as its own text block.
    """
    out: list[Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.kind == "caption":
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is not None and nxt.kind in ("table", "figure"):
                out.append(_merge_into(nxt, block, caption_first=True))
                i += 2
                continue
            if out and out[-1].kind in ("table", "figure"):
                out[-1] = _merge_into(out[-1], block, caption_first=False)
                i += 1
                continue
            block = block._replace(kind="text")  # orphan caption → plain text
        out.append(block)
        i += 1
    return out


def _blocks_from_parse_page(page) -> list[Block]:
    """Regions → blocks, folding captions into the visual they belong to."""
    figures_by_id = {f.region_id: f for f in page.figures}
    blocks: list[Block] = []
    for region in page.regions:
        rendered = render_region(region, figures_by_id, page.page_num)
        if not rendered:  # headers/footers and empty regions
            continue
        kind = (
            "caption"
            if region.region_type in _CAPTION_TYPES
            else _KIND_BY_REGION.get(region.region_type, "text")
        )
        blocks.append(Block(
            page_num=page.page_num,
            kind=kind,
            markdown=rendered,
            text=region.text or rendered,
            region_ids=(region.region_id,),
        ))
    return _fold_captions(blocks)


# Standalone text with no blank lines (pdfium emits one line per text row)
# regroups into blocks of roughly this many characters, split at line ends.
_TEXT_BLOCK_TARGET_CHARS = 1000


def _paragraphs(text: str) -> list[str]:
    """Split page text into paragraph-ish units.

    Blank lines are the primary boundary. PDF text layers often have none
    (every visual row is its own line) — those regroup runs of lines into
    ~_TEXT_BLOCK_TARGET_CHARS units so the segmenter gets real blocks to work with.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in normalized.split("\n\n") if p.strip()]

    out: list[str] = []
    for para in paras:
        if len(para) <= _TEXT_BLOCK_TARGET_CHARS * 2:
            out.append(para)
            continue
        current: list[str] = []
        size = 0
        for line in para.split("\n"):
            current.append(line)
            size += len(line) + 1
            if size >= _TEXT_BLOCK_TARGET_CHARS:
                out.append("\n".join(current).strip())
                current, size = [], 0
        if current:
            out.append("\n".join(current).strip())
    return [p for p in out if p]


def _blocks_from_text(page_num: int, text: str) -> list[Block]:
    """Standalone path: paragraph-ish units are the atomic blocks (no provenance)."""
    return [
        Block(page_num=page_num, kind="text", markdown=para, text=para, region_ids=())
        for para in _paragraphs(text)
    ]


def extract_split_pages(source: "ParseResult | Path | str") -> list[SplitPage]:
    """Normalize either input into per-page block lists, applying the 500-page cap."""
    if isinstance(source, ParseResult):
        pages = [
            SplitPage(
                page_num=p.page_num,
                text=p.markdown or p.text or p.native_text,
                blocks=_blocks_from_parse_page(p),
            )
            for p in source.pages
        ]
    else:
        path = Path(source)
        fmt = detect_format(path)
        loaded, _ = load_pdf_content(path) if fmt == "pdf" else load_office_content(path)
        logger.info("split standalone load: %s (%d pages, no OCR)", path.name, len(loaded))
        pages = [
            SplitPage(page_num=i, text=cp.text, blocks=_blocks_from_text(i, cp.text))
            for i, cp in enumerate(loaded, start=1)
        ]

    if len(pages) > MAX_PAGES:
        logger.warning("document has %d pages — splitting the first %d only", len(pages), MAX_PAGES)
        pages = pages[:MAX_PAGES]
    return pages
