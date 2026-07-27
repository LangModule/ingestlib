"""backfill() / abackfill() — rebuild a vector store from stored artifacts.

Artifacts are the source of truth; the vector store is an index over them.
Backfill re-embeds every document's stored split chunks and upserts them —
parse/classify/split are REUSED from the artifact store, so no OCR server
is needed and a corpus re-indexes in embedding time, not pipeline time.

When to reach for it: switching embedding_provider (new vector space),
pointing at a new vector_store connector, or rebuilding a wiped index.
Upserts are idempotent, so backfilling an already-populated store is safe.
"""
import asyncio
import time

from ingestlib.services.ingest.ingestor import _embed_chunks
from ingestlib.services.lifecycle.models import BackfillResult
from ingestlib.storage import VectorStore, artifacts, default_store
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)


async def abackfill(
    *,
    store: VectorStore | None = None,
    namespace: str = "",
) -> BackfillResult:
    """Re-embed every stored document's chunks into a vector store (async).

    store     — target connector; defaults to config.yaml's selection
    namespace — which corpus partition to rebuild; documents keep the
                namespace they were ingested into
    """
    t0 = time.perf_counter()
    store = store or default_store()

    metas = await asyncio.to_thread(artifacts.list_documents)
    metas = [m for m in metas if m.namespace == namespace]

    documents = chunks_total = 0
    skipped: list[str] = []
    for meta in metas:
        try:
            split = await asyncio.to_thread(artifacts.load_split, meta.doc_id)
        except FileNotFoundError:
            logger.warning(
                "backfill: %s (%s) has no split artifact — needs a real "
                "ingest, skipping", meta.doc_id[:12], meta.filename or "?",
            )
            skipped.append(meta.doc_id)
            continue
        chunks = split.chunks
        if not chunks:
            continue
        embeddings = await _embed_chunks([c.embedding_text for c in chunks])
        await asyncio.to_thread(
            store.upsert_chunks,
            meta.doc_id, chunks, embeddings,
            category=meta.category, namespace=namespace,
        )
        documents += 1
        chunks_total += len(chunks)
        logger.info(
            "backfilled %s: %d chunk(s) (%s)",
            meta.filename or meta.doc_id[:12], len(chunks), meta.category or "?",
        )

    result = BackfillResult(
        documents=documents,
        chunks=chunks_total,
        skipped=skipped,
        duration_seconds=round(time.perf_counter() - t0, 2),
    )
    logger.info(
        "backfill done: %d document(s), %d chunk(s) into %s",
        result.documents, result.chunks, type(store).__name__,
    )
    return result


def backfill(
    *,
    store: VectorStore | None = None,
    namespace: str = "",
) -> BackfillResult:
    """Re-embed every stored document's chunks into a vector store. Sync
    wrapper — use abackfill() inside an event loop."""
    return run_sync(abackfill(store=store, namespace=namespace), "abackfill")
