"""Retrieve service — question in, ranked cited results out.

    from ingestlib.services import retrieve
    result = retrieve("how were participants recruited?")
    print(result.context)          # numbered, cited chunks for an LLM prompt

    # documents + databases behind one call (see sources.example.yaml):
    result = retrieve("is rx 4471 ready?", sources=["prescriptions", "inserts"])
"""
from ingestlib.services.retrieve.models import Hit, RetrievalResult
from ingestlib.services.retrieve.retriever import aretrieve, retrieve
from ingestlib.sources.base import SourceResult

__all__ = ["retrieve", "aretrieve", "RetrievalResult", "Hit", "SourceResult"]
