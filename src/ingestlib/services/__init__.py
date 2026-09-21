"""User-facing services — the tools composed into complete flows.

    from ingestlib.services import ingest, retrieve

    ingest("report.pdf")                            # document → searchable chunks
    result = retrieve("what was Q1 revenue?")       # question → ranked cited chunks

ingest runs parse → classify → split → embed → vector upsert, recording every
stage's queryable output in the registry (the hub) and its bytes in the blob
store. retrieve runs embed → vector search → rerank (config.yaml's `reranker`
key: jina | aws | none), returning hits with full source provenance.
"""
from ingestlib.services.document import (
    StoredDocument,
    aget_document,
    get_document,
)
from ingestlib.services.extract import aextract, extract
from ingestlib.services.ingest import IngestResult, StageCallback, aingest, ingest
from ingestlib.services.lifecycle import (
    RecollectResult,
    ReindexResult,
    RemoveResult,
    SyncAction,
    SyncResult,
    arecollect,
    aremove,
    areindex,
    async_sync,
    recollect,
    reindex,
    remove,
    sync,
)
from ingestlib.services.retrieve import (
    Hit,
    RetrievalResult,
    SourceResult,
    aretrieve,
    retrieve,
)
from ingestlib.services.verify import (
    VerifyItem,
    VerifyResult,
    averify,
    verify,
)

__all__ = [
    "ingest",
    "aingest",
    "IngestResult",
    "StageCallback",
    "get_document",
    "aget_document",
    "StoredDocument",
    "extract",
    "aextract",
    "retrieve",
    "aretrieve",
    "RetrievalResult",
    "Hit",
    "SourceResult",
    "remove",
    "aremove",
    "RemoveResult",
    "sync",
    "async_sync",
    "SyncResult",
    "SyncAction",
    "reindex",
    "areindex",
    "ReindexResult",
    "recollect",
    "arecollect",
    "RecollectResult",
    "verify",
    "averify",
    "VerifyResult",
    "VerifyItem",
]
