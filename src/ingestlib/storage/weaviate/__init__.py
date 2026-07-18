"""Weaviate connector — HNSW dense + native BM25 hybrid, fused server-side.

Works against a local server (docker, HTTP and gRPC ports published) or
Weaviate Cloud. One endpoint (WEAVIATE_URL, plus WEAVIATE_API_KEY for
secured deployments) — the collection bootstraps on first use.
"""
from ingestlib.storage.weaviate.client import (
    ensure_collection,
    get_weaviate_client,
    reset_weaviate_client,
)
from ingestlib.storage.weaviate.store import WeaviateStore

__all__ = [
    "WeaviateStore",
    "get_weaviate_client",
    "reset_weaviate_client",
    "ensure_collection",
]
