"""Pinecone backend — VectorStore connector on a serverless index.

The index is created automatically on first upsert (name/cloud/region from
config.yaml, API key from PINECONE_API_KEY in .env).
"""
from ingestlib.storage.pinecone.client import (
    ensure_index,
    get_pinecone_client,
    reset_pinecone_client,
)
from ingestlib.storage.pinecone.store import PineconeStore

__all__ = [
    "PineconeStore",
    "get_pinecone_client",
    "reset_pinecone_client",
    "ensure_index",
]
