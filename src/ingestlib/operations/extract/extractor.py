"""extract() / aextract() — schema-driven field extraction with provenance.

The caller's Pydantic schema goes in; validated instances come out, each
field carrying the parse regions it was read from, a grounding verdict, and
a verification-capped confidence. mode="one" extracts a single instance per
document (an invoice, a contract); mode="many" extracts every instance (all
receipts in a scan bundle, all line items).

Independent of parse in the same way classify is: a ParseResult gives
region-level provenance; a raw path reads native text with page-level
provenance (scans must parse() first — same guard as everywhere).
"""
import asyncio
import time
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, Field, create_model

from ingestlib.foundations.llm import Image as LLMImage
from ingestlib.foundations.llm import achat_structured
from ingestlib.operations.extract.context import (
    SourcePage,
    build_pages,
    format_window,
    prepare_windows,
)
from ingestlib.operations.extract.grounding import assess_item
from ingestlib.operations.extract.models import ExtractedItem, ExtractResult, FieldValue
from ingestlib.operations.parse.models import ParseResult
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_LLM_CONCURRENCY = 8
_WINDOW_IMAGES = 4

_SYSTEM_PROMPT = (
    "You are a document data-extraction engine. Extract ONLY values that "
    "appear in the provided document text — never invent, infer beyond the "
    "text, or fill from general knowledge. Every block of text is tagged "
    "with its citation marker, like [p4:r2]. For EVERY field you fill, "
    "report the marker(s) of the block(s) the VALUE itself appears in — "
    "copy them exactly, e.g. sources=['p4:r2']. If the text has no markers, "
    "cite the page alone, like 'p4'."
)


class _FieldEvidence(BaseModel):
    field: str = Field(description="a top-level field name from the schema")
    sources: list[str] = Field(
        description="where the value was read: ['p4:r2', ...] or ['p4'] "
                    "when no region markers exist"
    )
    # Optional on purpose: models frequently omit self-scores, and a missing
    # advisory number must not fail the whole extraction — grounding is the
    # real verification, and the caps in grounding.py still apply.
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


def _response_models(schema: Type[BaseModel]) -> tuple[type, type]:
    """Wrap the caller's schema into the shape the LLM must return."""
    item_model = create_model(
        f"_Extracted{schema.__name__}",
        data=(schema, ...),
        evidence=(list[_FieldEvidence], Field(
            default_factory=list,
            description="one entry per filled field, citing its sources",
        )),
    )
    many_model = create_model(
        f"_Extracted{schema.__name__}List",
        items=(list[item_model], Field(
            default_factory=list,
            description="one entry per distinct instance found; empty if none",
        )),
    )
    return item_model, many_model


def _window_images(window: list[SourcePage]) -> list[LLMImage]:
    images = [LLMImage(data, "png") for p in window for data in p.images]
    return images[:_WINDOW_IMAGES]


def _to_item(raw, window: list[SourcePage]) -> ExtractedItem:
    fields, pages = assess_item(raw.data, raw.evidence, window)
    return ExtractedItem(value=raw.data, fields=fields, pages=pages)


async def _extract_one_window(
    window: list[SourcePage],
    schema: Type[SchemaT],
    instructions: str | None,
) -> ExtractedItem:
    item_model, _ = _response_models(schema)
    prompt = (
        f"Extract the fields of ONE {schema.__name__} from this document."
        f"{_instruction_block(instructions)}\n\n{format_window(window)}"
    )
    raw = await achat_structured(
        prompt, item_model, images=_window_images(window), system=_SYSTEM_PROMPT
    )
    return _to_item(raw, window)


async def _extract_many_window(
    window: list[SourcePage],
    schema: Type[SchemaT],
    instructions: str | None,
    semaphore: asyncio.Semaphore,
) -> list[ExtractedItem]:
    _, many_model = _response_models(schema)
    prompt = (
        f"Extract EVERY distinct {schema.__name__} in this document excerpt — "
        f"one item per instance (e.g. one per receipt, invoice, or entry). "
        f"Do not merge separate instances. Return no items if none appear."
        f"{_instruction_block(instructions)}\n\n{format_window(window)}"
    )
    async with semaphore:
        raw = await achat_structured(
            prompt, many_model, images=_window_images(window), system=_SYSTEM_PROMPT
        )
    return [_to_item(r, window) for r in raw.items]


def _instruction_block(instructions: str | None) -> str:
    return f"\n\nDomain guidance: {instructions}" if instructions else ""


def _merge_one(items: list[ExtractedItem], schema: Type[SchemaT]) -> ExtractedItem:
    """Field-by-field merge of per-window extractions of the SAME instance:
    for each field, the window with the highest verified confidence wins."""
    if len(items) == 1:
        return items[0]
    merged_data: dict = {}
    merged_fields: dict[str, FieldValue] = {}
    for name in schema.model_fields:
        best = max(items, key=lambda it: it.fields[name].confidence)
        merged_data[name] = getattr(best.value, name)
        merged_fields[name] = best.fields[name]
    pages = sorted({p for f in merged_fields.values() for p in f.pages})
    return ExtractedItem(
        value=schema.model_validate(merged_data), fields=merged_fields, pages=pages
    )


def _dedup_many(items: list[ExtractedItem]) -> list[ExtractedItem]:
    """Overlapping windows see the same instance twice — identical extracted
    values are one instance. Differing extractions are both kept: dropping a
    near-duplicate silently would hide a real ambiguity."""
    seen: set[str] = set()
    unique: list[ExtractedItem] = []
    for item in items:
        key = item.value.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


async def aextract(
    source: ParseResult | Path | str,
    schema: Type[SchemaT],
    *,
    mode: str = "one",
    target_pages: str | None = None,
    instructions: str | None = None,
) -> ExtractResult:
    """Extract schema instances from a document (async).

    source       — a ParseResult from parse() (region-level provenance), or a
                   document path (native text, page-level provenance; scans
                   and images have no text layer — parse() them first)
    schema       — any Pydantic model; its top-level fields are what gets
                   extracted and cited
    mode         — "one": a single instance per document (invoice, contract);
                   "many": every instance found (receipts, line items)
    target_pages — optional 1-based page selection like "1,3,5-7"
    instructions — optional domain guidance appended to the prompt, e.g.
                   "totals include tax; dates as YYYY-MM-DD"

    Provenance is verified, never trusted: cited regions must exist, and a
    value that can't be found in its cited source has its confidence capped
    (FieldValue.grounded says which). 100-page cap, as everywhere.
    """
    if mode not in ("one", "many"):
        raise ValueError(f"mode must be 'one' or 'many', got {mode!r}")
    start = time.perf_counter()
    # loading is blocking work (pypdfium2, LibreOffice) — off the event loop
    pages = await asyncio.to_thread(build_pages, source)
    windows, pages_used = prepare_windows(pages, mode, target_pages)
    logger.info(
        "extract start: schema=%s mode=%s pages=%d window(s)=%d",
        schema.__name__, mode, pages_used, len(windows),
    )

    if mode == "one":
        if len(windows) == 1:
            items = [await _extract_one_window(windows[0], schema, instructions)]
        else:
            semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)

            async def one(w):
                async with semaphore:
                    return await _extract_one_window(w, schema, instructions)

            tasks = [asyncio.ensure_future(one(w)) for w in windows]
            try:
                per_window = list(await asyncio.gather(*tasks))
            except BaseException:
                for task in tasks:
                    task.cancel()
                raise
            items = [_merge_one(per_window, schema)]
    else:
        semaphore = asyncio.Semaphore(_LLM_CONCURRENCY)
        tasks = [
            asyncio.ensure_future(
                _extract_many_window(w, schema, instructions, semaphore)
            )
            for w in windows
        ]
        try:
            window_items = list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        items = _dedup_many([item for batch in window_items for item in batch])

    result = ExtractResult(
        items=items,
        schema_name=schema.__name__,
        mode=mode,
        pages_used=pages_used,
        duration_seconds=round(time.perf_counter() - start, 2),
    )
    logger.info(
        "extract done: %d %s item(s) in %.1fs",
        len(result.items), schema.__name__, result.duration_seconds,
    )
    return result


def extract(
    source: ParseResult | Path | str,
    schema: Type[SchemaT],
    *,
    mode: str = "one",
    target_pages: str | None = None,
    instructions: str | None = None,
) -> ExtractResult:
    """Extract schema instances from a document. Sync wrapper — use aextract()
    inside an event loop."""
    return run_sync(
        aextract(
            source, schema, mode=mode,
            target_pages=target_pages, instructions=instructions,
        ),
        "aextract",
    )
