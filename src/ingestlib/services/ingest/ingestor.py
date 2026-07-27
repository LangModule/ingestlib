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
from ingestlib.utils.sync import run_sync


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
    replaces: str | None = None,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    categories: dict[str, str] | None = None,
    target_pages: str | None = None,
    max_pages: int | None = None,
    vocabulary: dict[str, str] | None = None,
    unmatched: str | None = None,
    on_stage: StageCallback | None = None,
) -> IngestResult:
    """Run a document through the full pipeline (async).

    The content-rule arguments pass straight to classify and split with
    their exact semantics: None resolves from rules.yaml's preset, an
    explicit {} forces open-ended/discovery, explicit values always win.

    Lifecycle: a document's logical identity is (namespace, source path).
    When this path already holds an OLDER version (different checksum), the
    new version replaces it — the old version's vectors and artifacts are
    deleted AFTER the new one is fully live, so retrieval never has a gap
    (status="replaced"). Same checksum arriving from a new path is a move:
    only the registry re-points, nothing runs (status="moved").

    path             — PDF/DOCX/PPTX document, or a PNG/JPEG/WebP image
    store            — vector store connector; defaults to the one selected
                       by config.yaml's `vector_store` key
    namespace        — vector-store namespace for multi-corpus setups
    skip_existing    — return status="skipped" when this exact file (by
                       checksum) already completed the FULL pipeline; a run
                       that failed partway is retried. Dedup keys on file
                       CONTENT only — re-ingesting with different rules
                       still skips; pass skip_existing=False to re-run
    replaces         — explicit doc_id of the version this file supersedes;
                       normally unnecessary (path matching detects it), for
                       when the new version lives at a different path
    max_chunk_tokens — split's chunk-size ceiling
    categories       — classify rules {label: description}, max 20
    target_pages     — classify page selection like "1,3,5-7" (1-based)
    max_pages        — classify page cap applied after selection
    vocabulary       — split section categories {name: description}, max 50
    unmatched        — split's policy for pages fitting no category:
                       "other" | "require" | "skip"
    on_stage         — optional progress callback, called as on_stage(stage,
                       event) with stage parse|classify|split|embed|upsert —
                       plus replace when an old version is deleted — and
                       event start|done; a stage that raises leaves its
                       "start" unmatched. Exceptions from the callback are
                       logged and ignored. Never called on the skip_existing
                       fast path — nothing runs there.
    """
    path = Path(path)
    doc_id = await asyncio.to_thread(sha256_of_file, path)

    if replaces is not None and not await asyncio.to_thread(
        artifacts.document_exists, replaces
    ):
        raise ValueError(
            f"replaces={replaces[:12]!r}… matches no stored document — "
            f"list_documents() shows what's stored"
        )

    if skip_existing and await asyncio.to_thread(artifacts.ingest_complete, doc_id):
        meta = await asyncio.to_thread(artifacts.get_document_meta, doc_id)
        status = "skipped"
        resolved = str(path.resolve())
        if meta.source_path and meta.source_path != resolved:
            # THE MOVE CASE: same content, new location. The registry must
            # follow, or a later sync(old_dir, prune=True) would delete a
            # document that still exists.
            await asyncio.to_thread(artifacts.set_source_path, doc_id, path)
            status = "moved"
        logger.info(
            "ingest %s (already fully ingested): %s doc_id=%s",
            status, path.name, doc_id[:12],
        )
        return IngestResult(
            status=status,
            doc_id=doc_id,
            filename=path.name,
            category=meta.category,
            pages=meta.page_count,
            sections=meta.sections,
            chunks=meta.chunks,
        )

    # who does this version supersede? explicit replaces= wins; otherwise the
    # document currently claiming this (namespace, path)
    old_doc_id = replaces
    if old_doc_id is None:
        prior = await asyncio.to_thread(artifacts.find_by_path, path, namespace)
        if prior is not None and prior.doc_id != doc_id:
            old_doc_id = prior.doc_id
    if old_doc_id == doc_id:  # replacing a doc with its own bytes is a no-op
        old_doc_id = None

    store = store or default_store()
    durations: dict[str, float] = {}
    logger.info("ingest start: %s", path.name)

    # artifact saves and the vector upsert are sync network/disk calls —
    # keep them off the event loop
    with _stage("parse", durations, on_stage):
        parse_result = await aparse(path)
        await asyncio.to_thread(artifacts.save_parse, parse_result)

    with _stage("classify", durations, on_stage):
        classify_result = await aclassify(
            parse_result, categories,
            target_pages=target_pages, max_pages=max_pages,
        )
        await asyncio.to_thread(artifacts.save_classify, doc_id, classify_result)

    with _stage("split", durations, on_stage):
        split_result = await asplit(
            parse_result,
            category=classify_result.category,
            max_chunk_tokens=max_chunk_tokens,
            vocabulary=vocabulary,
            unmatched=unmatched,
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

    if old_doc_id is not None:
        # the new version is fully live (manifest written) — now the old one
        # goes. A crash here leaves BOTH versions (over-complete, sync()
        # repairs); the reverse order could lose the document entirely.
        from ingestlib.services.lifecycle.remover import aremove  # lazy: sync() imports aingest

        with _stage("replace", durations, on_stage):
            removed = await aremove(old_doc_id, namespace=namespace, store=store)
            logger.info(
                "replaced %s → %s (%d old vector(s) deleted)",
                old_doc_id[:12], doc_id[:12], removed.vectors_deleted,
            )

    result = IngestResult(
        status="replaced" if old_doc_id is not None else "ingested",
        doc_id=doc_id,
        filename=path.name,
        category=classify_result.category,
        confidence=classify_result.confidence,
        pages=parse_result.page_count,
        sections=len(split_result.sections),
        chunks=len(chunks),
        vectors=vectors,
        replaced_doc_id=old_doc_id or "",
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
    replaces: str | None = None,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    categories: dict[str, str] | None = None,
    target_pages: str | None = None,
    max_pages: int | None = None,
    vocabulary: dict[str, str] | None = None,
    unmatched: str | None = None,
    on_stage: StageCallback | None = None,
) -> IngestResult:
    """Run a document through the full pipeline. Sync wrapper — use aingest()
    inside an event loop."""
    return run_sync(
        aingest(
            path, store=store, namespace=namespace,
            skip_existing=skip_existing, replaces=replaces,
            max_chunk_tokens=max_chunk_tokens,
            categories=categories, target_pages=target_pages,
            max_pages=max_pages, vocabulary=vocabulary, unmatched=unmatched,
            on_stage=on_stage,
        ),
        "aingest",
    )
