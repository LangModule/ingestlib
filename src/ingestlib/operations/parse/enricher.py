"""Nova vision enrichment of visual regions — charts become data tables,
figures/diagrams become structured descriptions, and every visual region is
extracted as a PNG crop with its nearest caption attached.

The 0.9B OCR model detects charts perfectly but misreads their values; Nova
re-reads each crop and its output replaces the region content.
"""
import asyncio
import dataclasses
import time

from ingestlib.foundations.llm import Image as NovaImage
from ingestlib.foundations.llm import achat
from ingestlib.foundations.ocr.models import Region
from ingestlib.operations.parse.models import FigureImage
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Region types that get cropped and sent to Nova.
_VISUAL_TYPES = ("chart", "figure")

_CAPTION_TYPES = ("figure_caption", "table_caption")

# A caption counts as "attached" when its vertical gap to the visual region is
# below this fraction of the page height (and the two overlap horizontally).
_CAPTION_MAX_GAP_FRACTION = 0.06

_PROMPT = (
    "This image is a cropped region from a document page.\n"
    "If it is a data chart (bar, line, pie, combo): extract the underlying data as "
    "a markdown table with a short bold title. Use exact values where printed; "
    "prefix values you estimate from bar/line heights with ~. Also include any "
    "printed callouts, growth labels, or annotations shown on the chart — as a "
    "table column or a line below the table.\n"
    "If it is a diagram, illustration, photo, or logo (not a data chart): do NOT "
    "invent numbers — give a concise structured description of what it shows.\n"
    "Output only the table or description, no commentary."
)


def _horizontal_overlap(a: Region, b: Region) -> float:
    return min(a.bbox.x2, b.bbox.x2) - max(a.bbox.x, b.bbox.x)


def _vertical_gap(a: Region, b: Region) -> float:
    """Distance between the closest vertical edges of two regions (0 if they overlap)."""
    if a.bbox.y2 <= b.bbox.y:
        return b.bbox.y - a.bbox.y2
    if b.bbox.y2 <= a.bbox.y:
        return a.bbox.y - b.bbox.y2
    return 0.0


def find_caption(visual: Region, regions: list[Region], page_height: int) -> str:
    """Text of the caption region nearest to `visual`, "" when none qualifies.

    A caption qualifies when it overlaps the visual horizontally and sits within
    _CAPTION_MAX_GAP_FRACTION of the page height above or below it.
    """
    max_gap = page_height * _CAPTION_MAX_GAP_FRACTION
    best_text, best_gap = "", max_gap
    for c in regions:
        if c.region_type not in _CAPTION_TYPES or not c.text.strip():
            continue
        if _horizontal_overlap(visual, c) <= 0:
            continue
        gap = _vertical_gap(visual, c)
        if gap <= best_gap:
            best_text, best_gap = c.text.strip(), gap
    return best_text


async def _enrich_one(
    region: Region,
    page_image: bytes,
    caption: str,
    semaphore: asyncio.Semaphore,
) -> tuple[Region, FigureImage]:
    """Crop the region, send it to Nova, return (updated region, FigureImage)."""
    crop = region.bbox.crop(page_image)
    async with semaphore:
        t0 = time.perf_counter()
        description = (await achat(_PROMPT, images=[NovaImage(data=crop, format="png")])).strip()
        logger.info(
            "enriched region %d [%s]: %.1fs, %d chars",
            region.region_id, region.region_type, time.perf_counter() - t0, len(description),
        )
    updated = dataclasses.replace(region, text=description, content=description)
    figure = FigureImage(
        region_id=region.region_id,
        region_type=region.region_type,
        image_bytes=crop,
        caption=caption,
        description=description,
    )
    return updated, figure


async def enrich_page(
    regions: list[Region],
    page_image: bytes,
    page_height: int,
    semaphore: asyncio.Semaphore,
) -> tuple[list[Region], list[FigureImage]]:
    """Enrich every chart/figure region on a page via Nova, in parallel.

    Returns (regions with enriched content, extracted FigureImages). Pages with
    no visual regions return unchanged regions and an empty list — no Nova calls.
    """
    visuals = [r for r in regions if r.region_type in _VISUAL_TYPES]
    if not visuals:
        return regions, []

    results = await asyncio.gather(*[
        _enrich_one(r, page_image, find_caption(r, regions, page_height), semaphore)
        for r in visuals
    ])

    updated_by_id = {r.region_id: r for r, _ in results}
    merged = [updated_by_id.get(r.region_id, r) for r in regions]
    figures = [fig for _, fig in results]
    return merged, figures
