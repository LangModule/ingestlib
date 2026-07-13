"""Ingest service — one call from document to searchable, cited chunks.

    from ingestlib.services import ingest
    result = ingest("report.pdf")
"""
from ingestlib.services.ingest.ingestor import aingest, ingest
from ingestlib.services.ingest.models import IngestResult

__all__ = ["ingest", "aingest", "IngestResult"]
