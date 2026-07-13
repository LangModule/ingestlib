"""Storage backends for pipeline outputs.

Sub-packages / modules:
    s3        — Amazon S3 singleton client + bucket bootstrap
    artifacts — persist/load parse, classify, and split outputs on S3,
                keyed by document checksum
    base      — VectorStore contract + RetrievedChunk (works with any
                vector database via connectors)
    pinecone  — Pinecone serverless connector, hybrid dense + sparse
    qdrant    — Qdrant connector, dense (local docker or Qdrant Cloud)

Services pick their connector via default_store(), driven by config.yaml's
`vector_store` key — both providers' keys can sit in .env; only the selected
one ever builds a client.
"""
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.pinecone import PineconeStore
from ingestlib.storage.qdrant import QdrantStore
from ingestlib.storage.s3 import ensure_bucket, get_s3_client, reset_s3_client

_STORES: dict[str, type[VectorStore]] = {
    "pinecone": PineconeStore,
    "qdrant": QdrantStore,
}


def default_store() -> VectorStore:
    """The connector selected by config.yaml's `vector_store` key."""
    from ingestlib.config import get_config

    name = get_config().vector_store
    if name not in _STORES:
        raise ValueError(
            f"unknown vector_store {name!r} in config.yaml — "
            f"choose one of {sorted(_STORES)}"
        )
    return _STORES[name]()


__all__ = [
    "get_s3_client",
    "reset_s3_client",
    "ensure_bucket",
    "VectorStore",
    "RetrievedChunk",
    "PineconeStore",
    "QdrantStore",
    "default_store",
]
