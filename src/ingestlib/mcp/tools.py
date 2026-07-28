"""The MCP tool functions — thin async wrappers over ingestlib's services.

Each returns a small, JSON-serializable dict (snippets truncated, no raw
markdown blobs) so results fit an agent's context. No business logic lives
here — these call the same functions the CLI does. `WRITE_TOOLS` names the
destructive ones the server hides under `mcp.read_only`.
"""
import asyncio
from pathlib import Path
from typing import Any

from ingestlib.mcp.schema import model_from_json_schema
from ingestlib.operations import aclassify, aextract
from ingestlib.services import abackfill, aingest, aremove, aretrieve, async_sync
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
) -> dict[str, Any]:
    """Search the corpus for a question and return ranked, cited chunks.

    Hybrid dense + lexical retrieval with reranking. Each hit carries its
    citation (doc · page · section) so answers can point to the source.
    filters accepts equality constraints on document_id/category/section/kind.
    """
    result = await aretrieve(
        question, top_k=top_k, namespace=namespace, filters=filters, rerank=rerank
    )
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


async def backfill(namespace: str = "") -> dict[str, Any]:
    """Rebuild the vector store by re-embedding stored artifacts — no re-parse.

    For a provider switch, a new connector, or a wiped index."""
    r = await abackfill(namespace=namespace)
    return {"documents": r.documents, "chunks": r.chunks,
            "skipped": len(r.skipped)}


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


# every tool, and the corpus-modifying subset the server hides under read_only
# (search/extract/classify/list_documents/doctor only READ — always available)
ALL_TOOLS = (search, ingest, extract, classify, list_documents, remove, sync,
             backfill, doctor)
WRITE_TOOLS = frozenset({"ingest", "remove", "sync", "backfill"})
