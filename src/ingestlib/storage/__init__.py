"""Storage backends for pipeline outputs.

Sub-packages / modules:
    s3        — Amazon S3 singleton client + bucket bootstrap
    blobs     — artifact-store backends (S3 bucket or a local folder),
                selected by config.yaml's `artifact_store` key
    artifacts — persist/load parse, classify, and split outputs on the
                selected backend, keyed by document checksum
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
    weaviate  — Weaviate connector (HNSW dense + native BM25, fused
                server-side), hybrid — local docker or Weaviate Cloud

Services pick their connector via default_store(), driven by config.yaml's
`vector_store` key. Connectors load LAZILY: sqlite ships with the core
install, the server-backed SDKs are pip extras — nothing here imports one
until it's actually selected, and a missing SDK raises the exact
`pip install "ingestlib[...]"` to run.
"""
import importlib

from ingestlib.storage.base import RetrievedChunk, VectorStore

# store class name → (subpackage, pip extra; None = ships with core)
_STORE_CLASSES: dict[str, tuple[str, str | None]] = {
    "PineconeStore": ("pinecone", "pinecone"),
    "QdrantStore": ("qdrant", "qdrant"),
    "SqliteStore": ("sqlite", None),
    "PgvectorStore": ("pgvector", "pgvector"),
    "MongodbStore": ("mongodb", "mongodb"),
    "MilvusStore": ("milvus", "milvus"),
    "OpensearchStore": ("opensearch", "opensearch"),
    "WeaviateStore": ("weaviate", "weaviate"),
}

# config.yaml `vector_store` value → store class name
_BY_CONFIG_NAME = {
    "pinecone": "PineconeStore",
    "qdrant": "QdrantStore",
    "sqlite": "SqliteStore",
    "pgvector": "PgvectorStore",
    "mongodb": "MongodbStore",
    "milvus": "MilvusStore",
    "opensearch": "OpensearchStore",
    "weaviate": "WeaviateStore",
}

_S3_EXPORTS = ("get_s3_client", "reset_s3_client", "ensure_bucket")


def _load_store_class(class_name: str) -> type[VectorStore]:
    subpackage, extra = _STORE_CLASSES[class_name]
    try:
        module = importlib.import_module(f"ingestlib.storage.{subpackage}")
    except ModuleNotFoundError as exc:
        # A missing piece of ingestlib itself is a bug, not a missing extra.
        if extra is None or (exc.name or "").startswith("ingestlib"):
            raise
        raise ImportError(
            f"the {subpackage} connector needs its SDK (module {exc.name!r} is "
            f'not installed) — install it with: pip install "ingestlib[{extra}]"'
        ) from exc
    return getattr(module, class_name)


def __getattr__(name: str):
    if name in _STORE_CLASSES:
        return _load_store_class(name)
    if name in _S3_EXPORTS:
        from ingestlib.storage import s3

        return getattr(s3, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_store() -> VectorStore:
    """The connector selected by config.yaml's `vector_store` key."""
    from ingestlib.config import get_config

    name = get_config().vector_store
    if name not in _BY_CONFIG_NAME:
        raise ValueError(
            f"unknown vector_store {name!r} in config.yaml — "
            f"choose one of {sorted(_BY_CONFIG_NAME)}"
        )
    return _load_store_class(_BY_CONFIG_NAME[name])()


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
    "WeaviateStore",
    "default_store",
]
