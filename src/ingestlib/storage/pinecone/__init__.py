"""Pinecone backend — VectorStore connector on serverless indexes.

Hybrid by default: a dense index for embedding search and a sparse index for
lexical search, both created automatically on first upsert (names/cloud/region
from config.yaml, API key from PINECONE_API_KEY in .env).
"""
from ingestlib.storage.pinecone.client import (
    ensure_index,
    ensure_sparse_index,
    get_pinecone_client,
    reset_pinecone_client,
)
from ingestlib.storage.pinecone.store import PineconeStore

__all__ = [
    "PineconeStore",
    "get_pinecone_client",
    "reset_pinecone_client",
    "ensure_index",
    "ensure_sparse_index",
]
