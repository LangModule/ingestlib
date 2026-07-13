"""ingest() / aingest() — one call from document to searchable, cited chunks.

The full pipeline: parse → classify → split → embed → vector upsert, with
every stage's output persisted to the S3 artifact store and the vector sync
recorded in an ingest manifest. Documents are deduplicated by content
checksum — re-ingesting the same file is a no-op unless forced.
"""
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

from ingestlib.foundations.llm import aembed_text
from ingestlib.operations.classify import aclassify
from ingestlib.operations.parse import aparse
from ingestlib.operations.split import asplit
from ingestlib.services.ingest.models import IngestResult
from ingestlib.storage import PineconeStore, VectorStore, artifacts
from ingestlib.utils.files import sha256_of_file
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_EMBED_CONCURRENCY = 8


async def _embed_chunks(embedding_texts: list[str]) -> list[list[float]]:
    """Embed every chunk's contextualized text, bounded-parallel."""
    semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def one(text: str) -> list[float]:
        async with semaphore:
            return await aembed_text(text)

    return list(await asyncio.gather(*[one(t) for t in embedding_texts]))


async def aingest(
    path: Path | str,
    *,
    store: VectorStore | None = None,
    namespace: str = "",
    skip_existing: bool = True,
    max_chunk_tokens: int = 1024,
) -> IngestResult:
    """Run a document through the full pipeline (async).

    path             — PDF/DOCX/PPTX to ingest
    store            — vector store connector; defaults to PineconeStore()
    namespace        — vector-store namespace for multi-corpus setups
    skip_existing    — return status="skipped" when this exact file (by
                       checksum) was already ingested
    max_chunk_tokens — split's chunk-size ceiling
    """
    path = Path(path)
    doc_id = sha256_of_file(path)

    if skip_existing and artifacts.document_exists(doc_id):
        meta = artifacts.get_document_meta(doc_id)
        logger.info("ingest skipped (already present): %s doc_id=%s", path.name, doc_id[:12])
        return IngestResult(
            status="skipped",
            doc_id=doc_id,
            filename=path.name,
            category=meta.category,
            pages=meta.page_count,
            sections=meta.sections,
            chunks=meta.chunks,
        )

    store = store or PineconeStore()
    durations: dict[str, float] = {}
    logger.info("ingest start: %s", path.name)

    t0 = time.perf_counter()
    parse_result = await aparse(path)
    artifacts.save_parse(parse_result)
    durations["parse"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    classify_result = await aclassify(parse_result)
    artifacts.save_classify(doc_id, classify_result)
    durations["classify"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    split_result = await asplit(
        parse_result,
        category=classify_result.category,
        max_chunk_tokens=max_chunk_tokens,
    )
    artifacts.save_split(doc_id, split_result)
    durations["split"] = time.perf_counter() - t0

    chunks = split_result.chunks
    t0 = time.perf_counter()
    embeddings = await _embed_chunks([c.embedding_text for c in chunks])
    durations["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vectors = 0
    if chunks:  # an empty/blank document has nothing to vectorize
        vectors = store.upsert_chunks(
            doc_id, chunks, embeddings,
            category=classify_result.category, namespace=namespace,
        )
    else:
        logger.warning("document produced no chunks — nothing upserted: %s", path.name)
    artifacts.save_ingest_manifest(doc_id, {
        "store": type(store).__name__,
        "namespace": namespace,
        "dimension": len(embeddings[0]) if embeddings else 0,
        "vector_ids": [f"{doc_id}:{c.chunk_id}" for c in chunks],
        "category": classify_result.category,
        "embedded_at": datetime.now(timezone.utc).isoformat(),
    })
    durations["upsert"] = time.perf_counter() - t0

    result = IngestResult(
        status="ingested",
        doc_id=doc_id,
        filename=path.name,
        category=classify_result.category,
        confidence=classify_result.confidence,
        pages=parse_result.page_count,
        sections=len(split_result.sections),
        chunks=len(chunks),
        vectors=vectors,
        durations={k: round(v, 2) for k, v in durations.items()},
    )
    logger.info(
        "ingest done: %s → %s, %d chunk(s) in %.1fs",
        path.name, result.category, result.chunks, result.total_seconds,
    )
    return result


def ingest(
    path: Path | str,
    *,
    store: VectorStore | None = None,
    namespace: str = "",
    skip_existing: bool = True,
    max_chunk_tokens: int = 1024,
) -> IngestResult:
    """Run a document through the full pipeline. Sync wrapper — use aingest()
    inside an event loop."""
    return asyncio.run(aingest(
        path, store=store, namespace=namespace,
        skip_existing=skip_existing, max_chunk_tokens=max_chunk_tokens,
    ))
