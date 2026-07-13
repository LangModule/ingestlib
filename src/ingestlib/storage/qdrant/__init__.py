"""Qdrant backend — VectorStore connector on a Qdrant collection.

The collection is created automatically on first upsert (name from
config.yaml, endpoint/key from QDRANT_URL and QDRANT_API_KEY in .env).
Works against local docker or Qdrant Cloud.
"""
from ingestlib.storage.qdrant.client import (
    ensure_collection,
    get_qdrant_client,
    reset_qdrant_client,
)
from ingestlib.storage.qdrant.store import QdrantStore

__all__ = [
    "QdrantStore",
    "get_qdrant_client",
    "reset_qdrant_client",
    "ensure_collection",
]
