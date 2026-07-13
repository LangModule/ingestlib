"""Retrieve service — question in, ranked cited chunks out.

    from ingestlib.services import retrieve
    result = retrieve("how were participants recruited?")
    print(result.context)          # numbered, cited chunks for an LLM prompt
"""
from ingestlib.services.retrieve.models import Hit, RetrievalResult
from ingestlib.services.retrieve.retriever import aretrieve, retrieve

__all__ = ["retrieve", "aretrieve", "RetrievalResult", "Hit"]
