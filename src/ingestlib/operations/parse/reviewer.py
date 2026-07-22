"""LLM page review — surgical per-region corrections, not a page rewrite.

The model sees the page image, the native text layer, and every region's current
content, and returns corrections keyed by region_id. Corrections apply to
individual regions so the markdown ↔ bbox mapping survives review untouched.
"""
import asyncio
import dataclasses
import json
import re
import time

from ingestlib.foundations.llm import Image as LLMImage
from ingestlib.foundations.llm import achat
from ingestlib.foundations.ocr.models import Region
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Regions the reviewer is asked to check. Visual regions are excluded — the enricher
# already produced their content from the same page with the same model.
_REVIEWABLE_TYPES = (
    "title", "text", "table", "formula", "reference",
    "figure_caption", "table_caption", "seal",
)

# Per-region content cap in the prompt; native-text cap for the whole page.
_REGION_CONTENT_LIMIT = 1500
_NATIVE_TEXT_LIMIT = 8000

_SYSTEM_PROMPT = """\
You are a document OCR reviewer. You receive a page image, the page's native
text layer (ground truth for character accuracy when present), and numbered
blocks of extracted content.

Compare each block against the page image and native text. Return a JSON array
of corrections — ONLY for blocks with real errors (misread characters, wrong
numbers, broken table structure, garbled words):

[{"region_id": <int>, "content": "<corrected block content>"}]

Rules:
- Return [] when every block is accurate. Most pages need no corrections.
- Keep each block's format: HTML stays HTML, LaTeX stays LaTeX, text stays text.
- Never add content that is not visible on the page.
- Output ONLY the JSON array, no commentary, no code fences."""

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _format_blocks(regions: list[Region]) -> tuple[str, set[int]]:
    """Prompt listing of reviewable blocks + the region_ids that were truncated.

    Truncated regions must not accept corrections — the model only saw a prefix, so
    applying its "corrected" block would silently cut the stored content.
    """
    lines: list[str] = []
    truncated: set[int] = set()
    for r in regions:
        content = (r.content or r.text or "").strip()
        if not content:
            continue
        if len(content) > _REGION_CONTENT_LIMIT:
            content = content[: _REGION_CONTENT_LIMIT - 3] + "..."
            truncated.add(r.region_id)
        lines.append(f"[region {r.region_id} | {r.region_type}]\n{content}")
    return "\n\n".join(lines), truncated


def _parse_corrections(response: str) -> dict[int, str]:
    """Response JSON → {region_id: corrected content}. Empty dict on bad JSON."""
    try:
        items = json.loads(_strip_fence(response.strip()))
    except json.JSONDecodeError:
        logger.warning("review response was not valid JSON, skipping corrections")
        return {}
    if not isinstance(items, list):
        return {}
    out: dict[int, str] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("region_id"), int):
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                out[item["region_id"]] = content.strip()
    return out


async def review_page(
    regions: list[Region],
    page_image: bytes,
    native_text: str,
    semaphore: asyncio.Semaphore,
) -> list[Region]:
    """Ask the LLM to verify the page's extracted content against the page image.

    Returns the regions list with corrections applied. Pages with nothing
    reviewable come back unchanged without an LLM call.
    """
    reviewable = [r for r in regions if r.region_type in _REVIEWABLE_TYPES]
    blocks, truncated = _format_blocks(reviewable)
    if not blocks:
        return regions

    native = native_text.strip()[:_NATIVE_TEXT_LIMIT] or "(none — scanned or image input)"
    user_prompt = (
        f"NATIVE TEXT LAYER:\n{native}\n\n"
        f"EXTRACTED BLOCKS:\n{blocks}\n\n"
        "Review the blocks against the page image and native text. "
        "Return the JSON array of corrections."
    )

    async with semaphore:
        t0 = time.perf_counter()
        response = await achat(
            user_prompt,
            images=[LLMImage(data=page_image, format="png")],
            system=_SYSTEM_PROMPT,
        )
    corrections = _parse_corrections(response)
    dropped = truncated & corrections.keys()
    if dropped:
        logger.warning(
            "dropping correction(s) for truncated region(s) %s — the model only saw "
            "a %d-char prefix, applying would cut the stored content",
            sorted(dropped), _REGION_CONTENT_LIMIT,
        )
        corrections = {k: v for k, v in corrections.items() if k not in dropped}
    logger.info(
        "review done: %.1fs, %d correction(s) across %d block(s)",
        time.perf_counter() - t0, len(corrections), len(reviewable),
    )

    if not corrections:
        return regions
    return [_apply_correction(r, corrections) for r in regions]


def _apply_correction(region: Region, corrections: dict[int, str]) -> Region:
    if region.region_id not in corrections or region.region_type not in _REVIEWABLE_TYPES:
        return region
    corrected = corrections[region.region_id]
    if region.region_type == "table":
        # content is HTML; keep the plain-text field free of markup.
        return dataclasses.replace(region, content=corrected)
    return dataclasses.replace(region, text=corrected, content=corrected)
