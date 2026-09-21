"""extract(source, schema) — schema-driven extraction, optionally persisted.

A thin wrapper over operations.extract that can PERSIST the ExtractResult to the
registry (the extractions table), so extractions become addressable and
retrievable via get_document — instead of a one-off call whose result is thrown
away. persist=True requires the document to already be in the corpus.
"""
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Type, TypeVar

from pydantic import BaseModel

from ingestlib.storage import registry
from ingestlib.utils.sync import run_sync

if TYPE_CHECKING:
    from ingestlib.operations.extract.models import ExtractResult
    from ingestlib.operations.parse.models import ParseResult

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _doc_id_of(source: "ParseResult | Path | str") -> str:
    from ingestlib.operations.parse.models import ParseResult

    if isinstance(source, ParseResult):
        if not source.source_checksum:
            raise ValueError("cannot persist extraction: the ParseResult has no source_checksum")
        return source.source_checksum
    from ingestlib.utils.files import sha256_of_file

    return sha256_of_file(Path(source))


async def aextract(
    source: "ParseResult | Path | str",
    schema: Type[SchemaT],
    *,
    mode: str = "one",
    target_pages: str | None = None,
    instructions: str | None = None,
    persist: bool = False,
) -> "ExtractResult":
    """Extract schema instances from a document (async); persist=True stores the
    result in the registry (the document must already be in the corpus)."""
    from ingestlib.operations.extract import aextract as _extract_op

    result = await _extract_op(
        source, schema, mode=mode, target_pages=target_pages, instructions=instructions,
    )
    if persist:
        doc_id = _doc_id_of(source)
        if not await asyncio.to_thread(registry.document_exists, doc_id):
            raise ValueError(
                f"cannot persist extraction: document {doc_id[:12]!r}… is not in the "
                f"corpus — ingest it first"
            )
        await asyncio.to_thread(registry.save_extraction, doc_id, result)
    return result


def extract(
    source: "ParseResult | Path | str",
    schema: Type[SchemaT],
    *,
    mode: str = "one",
    target_pages: str | None = None,
    instructions: str | None = None,
    persist: bool = False,
) -> "ExtractResult":
    """Extract schema instances from a document. Sync wrapper — use aextract()
    inside an event loop."""
    return run_sync(
        aextract(
            source, schema, mode=mode, target_pages=target_pages,
            instructions=instructions, persist=persist,
        ),
        "aextract",
    )
