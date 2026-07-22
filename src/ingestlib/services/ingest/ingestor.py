"""ingest() / aingest() — one call from document to searchable, cited chunks.

The full pipeline: parse → classify → split → embed → vector upsert, with
every stage's output persisted to the artifact store and the vector sync
recorded in an ingest manifest. Documents are deduplicated by content
checksum — re-ingesting the same file is a no-op unless forced.
"""
import asyncio
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from ingestlib.foundations.llm import aembed_text
from ingestlib.operations.classify import aclassify
from ingestlib.operations.parse import aparse
from ingestlib.operations.split import asplit
from ingestlib.operations.split.splitter import DEFAULT_MAX_CHUNK_TOKENS
from ingestlib.services.ingest.models import IngestResult
from ingestlib.storage import VectorStore, artifacts, default_store
from ingestlib.utils.files import sha256_of_file
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_EMBED_CONCURRENCY = 8

# on_stage(stage, event): stage ∈ parse|classify|split|embed|upsert, event ∈ start|done
StageCallback = Callable[[str, str], None]


def _notify(on_stage: StageCallback | None, stage: str, event: str) -> None:
    """Invoke the caller's progress callback; its bugs must never kill an ingest."""
    if on_stage is None:
        return
    try:
        on_stage(stage, event)
    except Exception:
        logger.warning("on_stage callback raised for (%s, %s) — ignored", stage, event)


@contextmanager
def _stage(
    name: str, durations: dict[str, float], on_stage: StageCallback | None
) -> Iterator[None]:
    """Time one pipeline stage and report its start/done to on_stage."""
    _notify(on_stage, name, "start")
    t0 = time.perf_counter()
    yield
    durations[name] = time.perf_counter() - t0
    _notify(on_stage, name, "done")


async def _embed_chunks(embedding_texts: list[str]) -> list[list[float]]:
    """Embed every chunk's contextualized text, bounded-parallel."""
    semaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)

    async def one(text: str) -> list[float]:
        async with semaphore:
            return await aembed_text(text)

    tasks = [asyncio.ensure_future(one(t)) for t in embedding_texts]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:  # don't leave sibling embed calls running
            task.cancel()
        raise


async def aingest(
    path: Path | str,
    *,
    store: VectorStore | None = None,
    namespace: str = "",
    skip_existing: bool = True,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    on_stage: StageCallback | None = None,
) -> IngestResult:
    """Run a document through the full pipeline (async).

    The classify stage honors rules.yaml's `classify:` preset (closed-set
    rules + page settings) when one exists — see rules.example.yaml.

    path             — PDF/DOCX/PPTX to ingest
    store            — vector store connector; defaults to the one selected
                       by config.yaml's `vector_store` key
    namespace        — vector-store namespace for multi-corpus setups
    skip_existing    — return status="skipped" when this exact file (by
                       checksum) already completed the FULL pipeline; a run
                       that failed partway is retried
    max_chunk_tokens — split's chunk-size ceiling
    on_stage         — optional progress callback, called as on_stage(stage,
                       event) with stage parse|classify|split|embed|upsert and
                       event start|done; a stage that raises leaves its
                       "start" unmatched. Exceptions from the callback are
                       logged and ignored. Never called on the skip_existing
                       fast path — nothing runs there.
    """
    path = Path(path)
    doc_id = await asyncio.to_thread(sha256_of_file, path)

    if skip_existing and await asyncio.to_thread(artifacts.ingest_complete, doc_id):
        meta = await asyncio.to_thread(artifacts.get_document_meta, doc_id)
        logger.info(
            "ingest skipped (already fully ingested): %s doc_id=%s", path.name, doc_id[:12]
        )
        return IngestResult(
            status="skipped",
            doc_id=doc_id,
            filename=path.name,
            category=meta.category,
            pages=meta.page_count,
            sections=meta.sections,
            chunks=meta.chunks,
        )

    store = store or default_store()
    durations: dict[str, float] = {}
    logger.info("ingest start: %s", path.name)

    # artifact saves and the vector upsert are sync network/disk calls —
    # keep them off the event loop
    with _stage("parse", durations, on_stage):
        parse_result = await aparse(path)
        await asyncio.to_thread(artifacts.save_parse, parse_result)

    with _stage("classify", durations, on_stage):
        classify_result = await aclassify(parse_result)
        await asyncio.to_thread(artifacts.save_classify, doc_id, classify_result)

    with _stage("split", durations, on_stage):
        split_result = await asplit(
            parse_result,
            category=classify_result.category,
            max_chunk_tokens=max_chunk_tokens,
        )
        await asyncio.to_thread(artifacts.save_split, doc_id, split_result)

    chunks = split_result.chunks
    with _stage("embed", durations, on_stage):
        embeddings = await _embed_chunks([c.embedding_text for c in chunks])

    with _stage("upsert", durations, on_stage):
        vectors = 0
        if chunks:  # an empty/blank document has nothing to vectorize
            vectors = await asyncio.to_thread(
                store.upsert_chunks,
                doc_id, chunks, embeddings,
                category=classify_result.category, namespace=namespace,
            )
        else:
            logger.warning("document produced no chunks — nothing upserted: %s", path.name)
        await asyncio.to_thread(artifacts.save_ingest_manifest, doc_id, {
            "store": type(store).__name__,
            "namespace": namespace,
            "dimension": len(embeddings[0]) if embeddings else 0,
            "vector_ids": [f"{doc_id}:{c.chunk_id}" for c in chunks],
            "category": classify_result.category,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
        })

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
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    on_stage: StageCallback | None = None,
) -> IngestResult:
    """Run a document through the full pipeline. Sync wrapper — use aingest()
    inside an event loop."""
    return asyncio.run(aingest(
        path, store=store, namespace=namespace,
        skip_existing=skip_existing, max_chunk_tokens=max_chunk_tokens,
        on_stage=on_stage,
    ))
