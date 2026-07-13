"""User-facing services — the tools composed into complete flows.

    from ingestlib.services import ingest, retrieve

    ingest("report.pdf")                            # document → searchable chunks
    result = retrieve("what was Q1 revenue?")       # question → ranked cited chunks

ingest runs parse → classify → split → embed → vector upsert, persisting every
stage to the artifact store. retrieve runs embed → vector search → Jina rerank,
returning hits with full source provenance.
"""
from ingestlib.services.ingest import IngestResult, aingest, ingest
from ingestlib.services.retrieve import Hit, RetrievalResult, aretrieve, retrieve

__all__ = [
    "ingest",
    "aingest",
    "IngestResult",
    "retrieve",
    "aretrieve",
    "RetrievalResult",
    "Hit",
]
