"""ingest() / aingest() — one call from document to searchable, cited chunks.

The full pipeline: parse → classify → split → embed → vector upsert. Every
stage's queryable output lands in the registry (the hub); the blob store keeps
the bytes (source, page PNGs, document.md). Documents are deduplicated by
content checksum — re-ingesting the same file is a no-op unless forced.
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
from ingestlib.storage import VectorStore, artifacts, default_store, registry
from ingestlib.utils.files import sha256_of_file
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)

_EMBED_CONCURRENCY = 8

# on_stage(stage, event): stage ∈ parse|classify|split|embed|upsert|replace|extract, event ∈ start|done
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


def _declared_collections() -> dict[str, dict]:
    """The rules.yaml classify categories as plain collection rows for the registry."""
    from ingestlib.config import get_config

    return {
        name: {
            "description": rule.description,
            "extract_schema": rule.extract_schema,
            "auto_extract": rule.auto_extract,
        }
        for name, rule in get_config().classify.collections.items()
    }


def _auto_extract_rule(category: str):
    """The CollectionRule for a category when it declares auto_extract + a schema, else None."""
    from ingestlib.config import get_config

    rule = get_config().classify.collections.get(category)
    if rule is not None and rule.auto_extract and rule.extract_schema:
        return rule
    return None


async def _run_auto_extract(rule, parse_result, doc_id: str, name: str) -> None:
    """Run the collection's declared extraction and persist it. Best-effort: the
    document is already fully ingested and live, so a failing auto-extract logs
    a warning rather than unwinding a successful ingest."""
    from ingestlib.schema import model_from_json_schema
    from ingestlib.services.extract import aextract

    try:
        model = model_from_json_schema(rule.description or "extract", rule.extract_schema)
        await aextract(parse_result, model, persist=True)
    except Exception as exc:
        logger.warning("auto-extract failed for %s (doc %s): %s", name, doc_id[:12], exc)


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
        await asyncio.gather(*tasks, return_exceptions=True)
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
                       plus replace when an old version is deleted, and extract
                       when the matched collection declares auto_extract — and
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
    await asyncio.to_thread(registry.sync_collections, _declared_collections())

    # artifact saves and the vector upsert are sync network/disk calls —
    # keep them off the event loop
    with _stage("parse", durations, on_stage):
        parse_result = await aparse(path)
        await asyncio.to_thread(artifacts.save_parse, parse_result)  # source + PNGs + document.md
        await asyncio.to_thread(
            registry.save_parse, doc_id, parse_result,
            namespace=namespace, source_path=path,
        )

    with _stage("classify", durations, on_stage):
        classify_result = await aclassify(
            parse_result, categories,
            target_pages=target_pages, max_pages=max_pages,
        )
        await asyncio.to_thread(registry.save_classify, doc_id, classify_result)

    with _stage("split", durations, on_stage):
        split_result = await asplit(
            parse_result,
            category=classify_result.category,
            max_chunk_tokens=max_chunk_tokens,
            vocabulary=vocabulary,
            unmatched=unmatched,
        )
        await asyncio.to_thread(registry.save_split, doc_id, split_result)

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
            if vectors < len(chunks):
                # the store acknowledged fewer vectors than chunks handed to it —
                # a silent short write. Fail loud: the doc stays incomplete
                # (status never reaches 'ingested'), so a re-ingest retries it
                # rather than the corpus quietly carrying a half-indexed document.
                raise RuntimeError(
                    f"vector store confirmed {vectors} of {len(chunks)} chunk(s) for "
                    f"{path.name} (doc {doc_id[:12]}…) — refusing to mark it ingested"
                )
        else:
            logger.warning("document produced no chunks — nothing upserted: %s", path.name)
        await asyncio.to_thread(
            registry.save_embed, doc_id,
            vector_store=type(store).__name__,
            vector_dim=len(embeddings[0]) if embeddings else 0,
            vector_count=vectors,
            embedded_at=datetime.now(timezone.utc),
        )

    if old_doc_id is not None:
        # the new version is fully live (manifest written) — now the old one
        # goes. A crash here leaves BOTH versions (over-complete, sync()
        # repairs); the reverse order could lose the document entirely.
        from ingestlib.services.lifecycle.remover import aremove  # lazy: sync() imports aingest

        with _stage("replace", durations, on_stage):
            removed = await aremove(
                old_doc_id, namespace=namespace, store=store, tombstone=True
            )
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
    # the freshly-ingested doc is LIVE ('ingested') whether or not it replaced an
    # older version — replaced_doc_id records the lineage; the OLD doc is the
    # tombstone (status='replaced', set by aremove above).
    await asyncio.to_thread(
        registry.save_status, doc_id,
        status="ingested", replaced_doc_id=result.replaced_doc_id,
    )

    rule = _auto_extract_rule(classify_result.category)
    if rule is not None:
        with _stage("extract", durations, on_stage):
            await _run_auto_extract(rule, parse_result, doc_id, path.name)

    # event-driven registry backup: runs only if config.yaml enabled it AND a
    # threshold tripped; best-effort, never fails the ingest
    from ingestlib.services.maintenance import maybe_backup_registry

    await asyncio.to_thread(maybe_backup_registry)

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
