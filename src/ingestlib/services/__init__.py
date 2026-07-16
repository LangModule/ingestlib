"""User-facing services — the tools composed into complete flows.

    from ingestlib.services import ingest, retrieve

    ingest("report.pdf")                            # document → searchable chunks
    result = retrieve("what was Q1 revenue?")       # question → ranked cited chunks

ingest runs parse → classify → split → embed → vector upsert, persisting every
stage to the artifact store. retrieve runs embed → vector search → rerank
(config.yaml's `reranker` key: jina | aws | none), returning hits with full
source provenance.
"""
from ingestlib.services.ingest import IngestResult, StageCallback, aingest, ingest
from ingestlib.services.retrieve import Hit, RetrievalResult, aretrieve, retrieve

__all__ = [
    "ingest",
    "aingest",
    "IngestResult",
    "StageCallback",
    "retrieve",
    "aretrieve",
    "RetrievalResult",
    "Hit",
]
