"""verify() — audit the durability ledger: registry expectation vs live store.

For every live document (or one doc_id), compare what the registry expects —
its chunk_count, set when the document was split — against what the vector store
actually holds right now (count_vectors). A mismatch means vectors silently went
missing or doubled since ingest: a partial upsert the ingest path didn't catch,
an out-of-band deletion, a store rebuilt from a stale snapshot. The ingest path
fails loud on a short write at write time; verify is the after-the-fact audit
that catches drift the corpus accumulated later.
"""
import asyncio

from pydantic import BaseModel, ConfigDict, Field

from ingestlib.storage import VectorStore, artifacts, default_store, registry
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)


class VerifyItem(BaseModel):
    """One document's durability check across all three stores."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str = ""
    namespace: str = ""
    expected: int                            # registry chunk_count — vectors that SHOULD exist
    recorded: int | None                     # registry vector_count — what upsert confirmed at ingest
    actual: int                              # vector store count_vectors — what the store holds now
    missing_blobs: list[str] = Field(default_factory=list)  # essential blobs absent (source, document.md)
    repaired: bool = False                   # verify --repair re-embedded this doc's vectors

    @property
    def vectors_ok(self) -> bool:
        return self.actual == self.expected

    @property
    def ok(self) -> bool:
        return self.vectors_ok and not self.missing_blobs


class VerifyResult(BaseModel):
    """The corpus-wide durability audit."""

    model_config = ConfigDict(frozen=True)

    items: list[VerifyItem] = Field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.items)

    @property
    def drifted(self) -> list[VerifyItem]:
        return [i for i in self.items if not i.ok]

    @property
    def ok(self) -> bool:
        return not self.drifted


async def _repair_vectors(row: dict, store: VectorStore) -> int:
    """Re-embed a drifted document's registry chunks back into the store; return
    the live vector count afterward. Repairs vector drift (a lost/partial upsert);
    it cannot conjure missing source blobs — those are reported, not fixed."""
    from ingestlib.services.lifecycle.reindexer import areembed_document

    doc_id, ns = row["doc_id"], row.get("namespace") or ""
    try:
        await areembed_document(
            doc_id, category=row.get("category") or "", namespace=ns, store=store,
        )
    except FileNotFoundError:
        logger.warning("verify --repair: %s has no stored split to re-embed", doc_id[:12])
    return await asyncio.to_thread(store.count_vectors, doc_id, ns)


async def averify(
    doc_id: str | None = None,
    *,
    namespace: str | None = None,
    store: VectorStore | None = None,
    repair: bool = False,
) -> VerifyResult:
    """Audit durability across the registry, the vector store, and the blob store (async).

    doc_id     — check only this document; None audits every live document
    namespace  — restrict to one partition (None = all); ignored when doc_id is given
    store      — vector store connector; defaults to config.yaml's `vector_store`
    repair     — re-embed vector-drifted documents from their registry chunks and
                 re-check (missing source/markdown blobs can't be repaired this way)
    """
    store = store or default_store()

    if doc_id is not None:
        row = await asyncio.to_thread(registry.meta_row, doc_id)
        rows = [row] if row is not None and row.get("status") != "replaced" else []
    else:
        rows = await asyncio.to_thread(registry.meta_rows, namespace)

    items: list[VerifyItem] = []
    for row in rows:
        did, ns = row["doc_id"], row.get("namespace") or ""
        expected = row.get("chunk_count") or 0
        actual = await asyncio.to_thread(store.count_vectors, did, ns)
        missing = await asyncio.to_thread(
            artifacts.missing_blobs, did, row.get("filename") or ""
        )
        repaired = False
        if repair and actual != expected:
            actual = await _repair_vectors(row, store)
            repaired = True
        items.append(VerifyItem(
            doc_id=did,
            filename=row.get("filename") or "",
            namespace=ns,
            expected=expected,
            recorded=row.get("vector_count"),
            actual=actual,
            missing_blobs=missing,
            repaired=repaired,
        ))

    result = VerifyResult(items=items)
    if result.ok:
        logger.info("verify: %d document(s) checked, all durable", result.checked)
    else:
        logger.warning(
            "verify: %d of %d document(s) drifted", len(result.drifted), result.checked,
        )
    return result


def verify(
    doc_id: str | None = None,
    *,
    namespace: str | None = None,
    store: VectorStore | None = None,
    repair: bool = False,
) -> VerifyResult:
    """Audit durability across all three stores. Sync wrapper — use averify()
    inside an event loop."""
    return run_sync(averify(doc_id, namespace=namespace, store=store, repair=repair), "averify")
