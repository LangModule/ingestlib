"""reindex() / areindex() — rebuild a vector store from the registry.

The source bytes are the ultimate truth; the vector store is a derived index.
Reindex re-embeds every document's stored split chunks (read from the registry)
and upserts them — parse/classify/split are REUSED, so no OCR server is needed
and a corpus re-indexes in embedding time, not pipeline time.

When to reach for it: switching embedding_provider (new vector space), pointing
at a new vector_store connector, or rebuilding a wiped/drifted index. Upserts
are idempotent, so reindexing an already-populated store is safe.
"""
import asyncio
import time

from ingestlib.services.ingest.ingestor import _embed_chunks
from ingestlib.services.lifecycle.models import ReindexResult
from ingestlib.storage import VectorStore, artifacts, default_store
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)


async def areembed_document(
    doc_id: str, *, category: str = "", namespace: str = "", store: VectorStore | None = None
) -> int:
    """Re-embed one document's registry chunks into the vector store; return the
    chunk count upserted (0 when the split has no chunks). Raises FileNotFoundError
    when the document has no stored split. Shared by reindex (whole corpus) and
    verify --repair (only the drifted documents)."""
    store = store or default_store()
    split = await asyncio.to_thread(artifacts.load_split, doc_id)
    chunks = split.chunks
    if not chunks:
        return 0
    embeddings = await _embed_chunks([c.embedding_text for c in chunks])
    await asyncio.to_thread(
        store.upsert_chunks, doc_id, chunks, embeddings, category=category, namespace=namespace,
    )
    return len(chunks)


async def areindex(
    *,
    store: VectorStore | None = None,
    namespace: str = "",
) -> ReindexResult:
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
            n = await areembed_document(
                meta.doc_id, category=meta.category, namespace=namespace, store=store,
            )
        except FileNotFoundError:
            logger.warning(
                "reindex: %s (%s) has no stored split — needs a real "
                "ingest, skipping", meta.doc_id[:12], meta.filename or "?",
            )
            skipped.append(meta.doc_id)
            continue
        if not n:
            continue
        documents += 1
        chunks_total += n
        logger.info(
            "reindexed %s: %d chunk(s) (%s)",
            meta.filename or meta.doc_id[:12], n, meta.category or "?",
        )

    result = ReindexResult(
        documents=documents,
        chunks=chunks_total,
        skipped=skipped,
        duration_seconds=round(time.perf_counter() - t0, 2),
    )
    logger.info(
        "reindex done: %d document(s), %d chunk(s) into %s",
        result.documents, result.chunks, type(store).__name__,
    )
    return result


def reindex(
    *,
    store: VectorStore | None = None,
    namespace: str = "",
) -> ReindexResult:
    """Re-embed every stored document's chunks into a vector store. Sync
    wrapper — use areindex() inside an event loop."""
    return run_sync(areindex(store=store, namespace=namespace), "areindex")
