"""Storage backends for pipeline outputs.

Sub-packages / modules:
    s3        — Amazon S3 singleton client + bucket bootstrap
    artifacts — persist/load parse, classify, and split outputs on S3,
                keyed by document checksum
    base      — VectorStore contract + RetrievedChunk (works with any
                vector database via connectors)
    pinecone  — Pinecone serverless connector, hybrid dense + sparse
    qdrant    — Qdrant connector, hybrid dense + BM25 sparse fused with
                server-side RRF (local docker or Qdrant Cloud)
    sqlite    — embedded zero-infrastructure connector (sqlite-vec + FTS5),
                hybrid, no server or credentials — one local file
    pgvector  — Postgres connector (pgvector HNSW + built-in full-text),
                hybrid — one connection URL to the Postgres you already run
    mongodb   — MongoDB connector (Atlas Vector Search + Atlas Search BM25),
                hybrid — Atlas any tier, atlas-local docker, or 8.2+ mongot
    milvus    — Milvus connector (dense ANN + server-side BM25, RRF fused
                server-side), hybrid — local docker or Zilliz Cloud
    opensearch — OpenSearch connector (faiss k-NN + Lucene BM25, RRF fused
                client-side), hybrid — an Amazon OpenSearch domain signed
                with the aws profile, or a local docker server

Services pick their connector via default_store(), driven by config.yaml's
`vector_store` key — every provider's keys can sit in .env; only the selected
one ever builds a client.
"""
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.milvus import MilvusStore
from ingestlib.storage.mongodb import MongodbStore
from ingestlib.storage.opensearch import OpensearchStore
from ingestlib.storage.pgvector import PgvectorStore
from ingestlib.storage.pinecone import PineconeStore
from ingestlib.storage.qdrant import QdrantStore
from ingestlib.storage.s3 import ensure_bucket, get_s3_client, reset_s3_client
from ingestlib.storage.sqlite import SqliteStore

_STORES: dict[str, type[VectorStore]] = {
    "pinecone": PineconeStore,
    "qdrant": QdrantStore,
    "sqlite": SqliteStore,
    "pgvector": PgvectorStore,
    "mongodb": MongodbStore,
    "milvus": MilvusStore,
    "opensearch": OpensearchStore,
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
    "SqliteStore",
    "PgvectorStore",
    "MongodbStore",
    "MilvusStore",
    "OpensearchStore",
    "default_store",
]
