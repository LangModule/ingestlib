"""recollect() / arecollect() — re-sort the corpus into collections (no OCR).

Re-runs classify on each stored document, rebuilt text-only from the registry,
updating its category + collection, and refreshes the declared collections'
metadata (description/extract_schema/auto_extract) from rules.yaml. The reindex
sibling: cheap re-labeling without re-parsing, the operation you run when
rules.yaml's classify rules change.
"""
import asyncio
import time

from ingestlib.services.lifecycle.models import RecollectItem, RecollectResult
from ingestlib.storage import registry
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)


async def arecollect(
    *,
    namespace: str | None = None,
    categories: dict[str, str] | None = None,
    target_pages: str | None = None,
    max_pages: int | None = None,
) -> RecollectResult:
    """Re-classify every stored document from the registry and re-sort it (async).

    namespace  — which partition to re-sort; None (default) re-sorts every
                 partition (a rules change is usually corpus-wide)
    categories — classify rules (as classify()/ingest() take them); None uses
                 rules.yaml's `classify:` preset — pass the same rules you changed
    """
    from ingestlib.operations.classify import aclassify
    from ingestlib.services.ingest.ingestor import _declared_collections

    t0 = time.perf_counter()
    await asyncio.to_thread(registry.sync_collections, _declared_collections())
    rows = await asyncio.to_thread(registry.meta_rows, namespace)

    processed = 0
    changed: list[RecollectItem] = []
    for row in rows:
        doc_id = row["doc_id"]
        parse = await asyncio.to_thread(registry.reconstruct_parse, doc_id)
        if parse is None:  # nothing to classify (no pages recorded)
            continue
        old = row.get("category", "")
        result = await aclassify(
            parse, categories, target_pages=target_pages, max_pages=max_pages
        )
        await asyncio.to_thread(registry.save_classify, doc_id, result)
        processed += 1
        if result.category != old:
            changed.append(RecollectItem(
                doc_id=doc_id, filename=row.get("filename", ""),
                from_category=old, to_category=result.category,
            ))

    outcome = RecollectResult(
        recollected=processed, changed=changed,
        duration_seconds=round(time.perf_counter() - t0, 2),
    )
    logger.info("recollect done: %d re-classified, %d re-sorted", processed, len(changed))
    return outcome


def recollect(
    *,
    namespace: str | None = None,
    categories: dict[str, str] | None = None,
    target_pages: str | None = None,
    max_pages: int | None = None,
) -> RecollectResult:
    """Re-sort the corpus into collections. Sync wrapper — use arecollect()
    inside an event loop."""
    return run_sync(
        arecollect(
            namespace=namespace, categories=categories,
            target_pages=target_pages, max_pages=max_pages,
        ),
        "arecollect",
    )
