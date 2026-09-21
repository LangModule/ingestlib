"""remove() / aremove() — erase one document from both stores.

The one honest delete, torn down most-derived first: vectors, then the blob
objects under the document's prefix, then the registry row LAST — so a crash
can never leave ghost vectors whose authoritative record is already gone, and
a half-done delete still shows the document as live for verify/reindex to
recover. Accepts the file path (how users think) or a doc_id, full or a unique
prefix (how `ingestlib list` prints them).
"""
import asyncio
from pathlib import Path

from ingestlib.services.lifecycle.models import RemoveResult
from ingestlib.storage import VectorStore, artifacts, default_store
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)


def _resolve_target(target: str, namespace: str) -> str:
    """Turn a path or a (possibly shortened) doc_id into the stored doc_id.

    Paths win first — a stored source_path is absolute, so a hex doc_id can
    never collide with one. Then exact doc_id, then a unique prefix.
    """
    by_path = artifacts.find_by_path(target, namespace=namespace)
    if by_path is not None:
        return by_path.doc_id

    if artifacts.document_exists(target):
        return target

    matches = [m for m in artifacts.list_documents() if m.doc_id.startswith(target)]
    if len(matches) == 1:
        return matches[0].doc_id
    if len(matches) > 1:
        raise ValueError(
            f"doc_id prefix {target!r} is ambiguous — it matches "
            f"{len(matches)} documents; give more characters"
        )
    raise ValueError(
        f"{target!r} matches no stored document — not a known source path "
        f"and not a doc_id; list_documents() (or `ingestlib list`) shows "
        f"what's stored"
    )


def _remove(
    doc_id: str, namespace: str, store: VectorStore | None, tombstone: bool = False
) -> RemoveResult:
    from ingestlib.storage import registry

    meta = artifacts.get_document_meta(doc_id)

    # vectors first — deleted under the namespace the registry recorded
    vectors = 0
    if artifacts.ingest_complete(doc_id):
        store = store or default_store()
        vec_ns = meta.namespace or namespace
        vectors = store.delete_document(doc_id, namespace=vec_ns)

    objects = artifacts.delete_document(doc_id)
    if tombstone:
        registry.set_status(doc_id, "replaced")   # keep the row as a lineage tombstone
    else:
        registry.delete_document(doc_id)           # full delete — children cascade
    logger.info(
        "%s %s (%s): %d vector(s), %d artifact object(s)",
        "tombstoned" if tombstone else "removed",
        doc_id[:12], meta.filename or "?", vectors, objects,
    )
    return RemoveResult(
        doc_id=doc_id,
        filename=meta.filename,
        vectors_deleted=vectors,
        artifacts_deleted=objects,
    )


async def aremove(
    target: Path | str,
    *,
    namespace: str = "",
    store: VectorStore | None = None,
    tombstone: bool = False,
) -> RemoveResult:
    """Erase one document — vectors AND artifacts (async).

    target    — the document's source path, or its doc_id (full or a unique
                prefix as printed by `ingestlib list`)
    namespace — scopes path resolution; the vector deletion itself uses the
                namespace the registry recorded for the document
    store     — vector store connector; defaults to config.yaml's selection
    tombstone — keep the registry row as status='replaced' instead of deleting
                it (the replace path uses this for lineage); vectors + blobs
                are deleted either way
    """
    doc_id = await asyncio.to_thread(_resolve_target, str(target), namespace)
    return await asyncio.to_thread(_remove, doc_id, namespace, store, tombstone)


def remove(
    target: Path | str,
    *,
    namespace: str = "",
    store: VectorStore | None = None,
) -> RemoveResult:
    """Erase one document — vectors AND artifacts. Sync wrapper — use
    aremove() inside an event loop."""
    return run_sync(aremove(target, namespace=namespace, store=store), "aremove")
