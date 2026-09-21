"""The MCP tool functions — thin async wrappers over ingestlib's services.

Each returns a small, JSON-serializable dict (snippets truncated, no raw
markdown blobs) so results fit an agent's context. No business logic lives
here — these call the same functions the CLI does. `WRITE_TOOLS` names the
destructive ones the server hides under `mcp.read_only`.
"""
import asyncio
from pathlib import Path
from typing import Any

from ingestlib.schema import model_from_json_schema
from ingestlib.operations import aclassify, aextract
from ingestlib.services import (
    aingest,
    arecollect,
    areindex,
    aremove,
    aretrieve,
    async_sync,
    averify,
)
from ingestlib.storage import artifacts

_SNIPPET = 400            # per-hit / per-value text cap
_CONTEXT_CAP = 6000       # prompt-block cap for search


def _clip(text: str, n: int = _SNIPPET) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


async def search(
    question: str,
    top_k: int = 5,
    namespace: str = "",
    filters: dict[str, Any] | None = None,
    rerank: bool = True,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Search for a question and return ranked, cited results.

    Documents by default: hybrid dense + lexical retrieval with reranking; each
    hit carries its citation (doc · page · section). filters accepts equality
    constraints on document_id/category/section/kind.

    sources: names from sources.yaml (documents and/or SQL databases). When
    given, retrieval fans out over them and returns normalized `results` — each
    with source, source_type, and provenance (SQL results carry the exact SQL).
    """
    result = await aretrieve(
        question, top_k=top_k, namespace=namespace, filters=filters,
        rerank=rerank, sources=sources,
    )
    if sources:
        results = [
            {"rank": i, "source": r.source, "source_type": r.source_type,
             "content": _clip(r.content), "provenance": r.provenance}
            for i, r in enumerate(result.results, start=1)
        ]
        return {"question": question, "result_count": len(results),
                "results": results, "context": result.context[:_CONTEXT_CAP]}
    hits = [
        {
            "rank": i,
            "score": round(h.rerank_score if h.rerank_score is not None else h.vector_score, 4),
            "citation": h.citation,
            "document_id": h.chunk.document_id,
            "pages": h.chunk.pages,
            "section": h.chunk.section,
            "heading": h.chunk.heading,
            "snippet": _clip(h.chunk.text or h.chunk.markdown),
        }
        for i, h in enumerate(result.hits, start=1)
    ]
    return {"question": question, "hit_count": len(hits), "hits": hits,
            "context": result.context[:_CONTEXT_CAP]}


async def ingest(path: str, namespace: str = "") -> dict[str, Any]:
    """Ingest one document (PDF/DOCX/PPTX/image) into the corpus.

    Runs parse → classify → split → embed → upsert. Re-ingesting an edited
    file replaces the old version. For a whole folder, use `sync` instead.
    """
    if not Path(path).exists():
        raise ValueError(f"file not found: {path}")
    r = await aingest(path, namespace=namespace)
    return {
        "status": r.status, "doc_id": r.doc_id, "filename": r.filename,
        "category": r.category, "chunks": r.chunks, "vectors": r.vectors,
        "replaced_doc_id": r.replaced_doc_id,
    }


async def extract(
    path: str,
    json_schema: dict[str, Any],
    mode: str = "one",
    instructions: str = "",
) -> dict[str, Any]:
    """Extract structured fields from a document into a schema you define.

    json_schema is a JSON Schema object ({"type":"object","properties":{…}}).
    mode="one" fills a single record; mode="many" finds every instance. Each
    field reports its pages, whether it was grounded in the source, and an
    honest confidence.
    """
    if not Path(path).exists():
        raise ValueError(f"file not found: {path}")
    model = model_from_json_schema("ExtractSchema", json_schema)
    result = await aextract(
        path, model, mode=mode, instructions=instructions or None
    )
    items = []
    for item in result.items:
        value = item.value.model_dump(mode="json") if hasattr(item.value, "model_dump") else item.value
        items.append({
            "value": value,
            "citation": item.citation,
            "pages": item.pages,
            "fields": {
                name: {"grounded": fv.grounded, "confidence": round(fv.confidence, 2),
                       "pages": fv.pages}
                for name, fv in item.fields.items()
            },
        })
    return {"schema": result.schema_name, "mode": result.mode,
            "item_count": len(items), "items": items}


async def classify(path: str, categories: dict[str, str] | None = None) -> dict[str, Any]:
    """Classify a document's type — open-ended, or constrained to your own
    {label: description} categories. Returns the label, confidence, and reasoning."""
    if not Path(path).exists():
        raise ValueError(f"file not found: {path}")
    r = await aclassify(path, categories)
    return {"category": r.category, "confidence": round(r.confidence, 3),
            "reasoning": _clip(getattr(r, "reasoning", ""), 600)}


async def list_documents(namespace: str | None = None) -> dict[str, Any]:
    """List documents in the corpus. Omit namespace for all; pass one to scope.

    Returns each doc's id, filename, source path, namespace, pages, category,
    and chunk count."""
    docs = await asyncio.to_thread(artifacts.list_documents)
    if namespace is not None:
        docs = [d for d in docs if d.namespace == namespace]
    return {"count": len(docs), "documents": [
        {"doc_id": d.doc_id, "filename": d.filename, "source_path": d.source_path,
         "namespace": d.namespace, "pages": d.page_count, "category": d.category,
         "chunks": d.chunks}
        for d in docs
    ]}


async def get_document(doc_id: str) -> dict[str, Any]:
    """Fetch one stored document by doc_id — metadata, section names, persisted
    extractions, and the whole-document markdown. Resolves a search hit's
    citation to real content."""
    from ingestlib.services import aget_document

    doc = await aget_document(doc_id)
    if doc is None:
        return {"error": f"no document {doc_id!r} in the corpus"}
    markdown = await asyncio.to_thread(doc.markdown)
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "source_path": doc.source_path,
        "namespace": doc.namespace,
        "category": doc.category,
        "collection": doc.collection,
        "confidence": doc.classify_confidence,
        "status": doc.status,
        "pages": doc.page_count,
        "sections": [s["name"] for s in doc.sections],
        "chunks": doc.chunk_count,
        "extractions": [
            {"schema": e.get("schema_name"), "value": e.get("value")}
            for e in doc.extractions
        ],
        "markdown": _clip(markdown or "", 4000),
    }


async def collections(namespace: str | None = None) -> dict[str, Any]:
    """List the corpus collections and their document counts. Omit namespace for
    all partitions. A collection with an attached extract schema is flagged, and
    auto_extract=true means that schema is pulled from every doc on ingest."""
    from ingestlib.storage import registry

    rows = await asyncio.to_thread(registry.list_collections, namespace)
    declared = {c["name"]: c for c in await asyncio.to_thread(registry.collection_rules)}
    out = []
    for row in rows:
        name = row["collection"] or ""
        rule = declared.get(name) or {}
        out.append({
            "collection": name or "(uncategorized)",
            "count": row["count"],
            "has_schema": bool(rule.get("extract_schema")),
            "auto_extract": bool(rule.get("auto_extract")),
        })
    return {"count": len(out), "collections": out}


async def remove(target: str, namespace: str = "") -> dict[str, Any]:
    """Erase one document from both stores (vectors and artifacts).

    target is a source path or a doc_id (full or a unique prefix)."""
    r = await aremove(target, namespace=namespace)
    return {"doc_id": r.doc_id, "filename": r.filename,
            "vectors_deleted": r.vectors_deleted,
            "artifacts_deleted": r.artifacts_deleted}


async def sync(
    directory: str,
    prune: bool = False,
    dry_run: bool = False,
    namespace: str = "",
) -> dict[str, Any]:
    """Reconcile a folder with the corpus: new files ingest, edited replace,
    renamed move, and — with prune — deleted files are removed.

    Set dry_run=True to preview the plan without changing anything. prune is
    root-scoped and refuses to run on an empty scan."""
    r = await async_sync(directory, namespace=namespace, prune=prune, dry_run=dry_run)
    return {"directory": r.directory, "dry_run": r.dry_run, "counts": r.counts,
            "actions": [{"action": a.action, "path": a.path, "detail": a.detail}
                        for a in r.actions]}


async def reindex(namespace: str = "") -> dict[str, Any]:
    """Rebuild the vector store by re-embedding the registry's chunks — no re-parse.

    For a provider switch, a new connector, or a wiped index."""
    r = await areindex(namespace=namespace)
    return {"documents": r.documents, "chunks": r.chunks,
            "skipped": len(r.skipped)}


async def recollect(namespace: str | None = None) -> dict[str, Any]:
    """Re-classify the stored corpus and re-sort it into collections — no re-parse.

    Run after the classification rules change (rules.yaml): each document is
    re-classified from the registry (text-only, no OCR) and its category/collection
    updated. Omit namespace to re-sort every partition. Reports which documents moved."""
    r = await arecollect(namespace=namespace)
    return {
        "recollected": r.recollected,
        "changed": [
            {"doc_id": c.doc_id[:12], "filename": c.filename,
             "from": c.from_category, "to": c.to_category}
            for c in r.changed
        ],
    }


async def describe_schema(source: str) -> dict[str, Any]:
    """Auto-document a SQL source's tables (LLM-generated one-line hints from
    sampled rows) — the `tables:` hints that drive text2SQL accuracy. Returns a
    {table: description} map to review before pasting into sources.yaml; does not
    edit sources.yaml itself. `source` is a SQL source name from sources.yaml."""
    from ingestlib.cli.describe import _describe_all

    tables = await _describe_all(source)
    return {"source": source, "table_count": len(tables), "tables": tables}


async def verify(namespace: str | None = None) -> dict[str, Any]:
    """Audit durability across all three stores: each document's expected chunk
    count (registry) vs what the vector store holds, plus whether its essential
    blobs (source bytes, document.md) exist. Read-only — reports drifted documents;
    repair (re-embedding) is a write action, done from the CLI (ingestlib verify --repair)."""
    r = await averify(namespace=namespace)
    return {
        "checked": r.checked,
        "durable": r.ok,
        "drifted": [
            {"doc_id": i.doc_id[:12], "filename": i.filename,
             "expected": i.expected, "actual": i.actual,
             "missing_blobs": i.missing_blobs}
            for i in r.drifted
        ],
    }


async def doctor() -> dict[str, Any]:
    """Health-check the configured stack with real calls (LLM, embeddings,
    reranker, artifact store, vector store, OCR server). Returns per-check
    status: ok | warn | fail | skip."""
    from ingestlib.cli import doctor as _doc

    checks = (
        ("python", _doc.check_python), ("libreoffice", _doc.check_libreoffice),
        ("ocr_server", _doc.check_ocr_server), ("llm", _doc.check_llm),
        ("embeddings", _doc.check_embeddings), ("reranker", _doc.check_reranker),
        ("artifact_store", _doc.check_artifact_store),
        ("vector_store", _doc.check_vector_store),
    )
    results = []
    for name, fn in checks:
        try:
            status, detail = await asyncio.to_thread(fn)
        except Exception as exc:  # a check must never crash the tool
            status, detail = "fail", str(exc)
        results.append({"check": name, "status": status, "detail": _clip(detail, 300)})
    ok = all(r["status"] != "fail" for r in results)
    return {"healthy": ok, "checks": results}


async def registry_status() -> dict[str, Any]:
    """Report the internal registry database's reachability, revision, and whether
    its schema is up to date. Read-only — the registry is ingestlib's own metadata
    store (the corpus spine), not user data."""
    from ingestlib.cli.registry import (
        _alembic_config, _current_revision, _registry_tables, _target,
    )
    from ingestlib_registry.db import ping

    def _run() -> dict[str, Any]:
        target = _target()
        if not ping():
            return {"reachable": False, "target": target}
        from alembic.script import ScriptDirectory

        current = _current_revision()
        head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
        return {
            "reachable": True, "target": target, "initialized": current is not None,
            "revision": current, "head": head, "up_to_date": current == head,
            "tables": _registry_tables() if current else [],
        }

    return await asyncio.to_thread(_run)


async def registry_init() -> dict[str, Any]:
    """Create or upgrade the internal registry schema to the latest revision
    (Alembic migrations). Idempotent — a no-op when already current. Run once
    before the corpus path (ingest/search) on a fresh or upgraded registry."""
    from alembic import command

    from ingestlib.cli.registry import (
        _alembic_config, _current_revision, _registry_tables, _target,
    )
    from ingestlib_registry.db import ping

    def _run() -> dict[str, Any]:
        if not ping():
            raise RuntimeError(
                f"registry unreachable at {_target()} — start it and check INGESTLIB_REGISTRY_URL"
            )
        command.upgrade(_alembic_config(), "head")
        return {"target": _target(), "revision": _current_revision(), "tables": _registry_tables()}

    return await asyncio.to_thread(_run)


async def registry_backup() -> dict[str, Any]:
    """Back up the registry (pg_dump) into the artifact store and return the stored
    key + byte size. The manual counterpart to config.yaml's automatic backups."""
    from ingestlib.cli.registry import backup_registry

    key, size = await asyncio.to_thread(backup_registry)
    return {"key": key, "bytes": size}


async def registry_restore() -> dict[str, Any]:
    """DESTRUCTIVE: restore the registry from the latest backup — pg_restore --clean
    DROPS and rewrites every table, discarding all current registry state. Use only
    to recover a corrupted/wiped registry; the source bytes remain the ultimate
    truth (reindex/re-ingest can rebuild derived state)."""
    from ingestlib.cli.registry import restore_registry

    key = await asyncio.to_thread(restore_registry)
    if key is None:
        return {"restored": False, "error": "no registry backups found — run registry_backup first"}
    return {"restored": True, "from": key}


# every tool, and the corpus-modifying subset the server hides under read_only.
# The read tools (search/extract/classify/list_documents/get_document/collections/
# describe_schema/verify/registry_status/doctor) are always available; registry_init/
# backup/restore modify the registry (restore is destructive) and hide under read_only.
ALL_TOOLS = (search, ingest, extract, classify, list_documents, get_document,
             collections, describe_schema, remove, sync, reindex, recollect,
             verify, doctor, registry_status, registry_init, registry_backup,
             registry_restore)
WRITE_TOOLS = frozenset({
    "ingest", "remove", "sync", "reindex", "recollect",
    "registry_init", "registry_backup", "registry_restore",
})
