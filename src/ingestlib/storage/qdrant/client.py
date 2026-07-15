"""Process-wide singleton Qdrant client + first-time collection bootstrap.

Works against a local server (docker run -p 6333:6333 qdrant/qdrant, no API
key) or Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY in .env).

One collection holds both retrieval signals as named vectors: "dense" (cosine,
dimension from the first embedding batch) and "sparse" (BM25 lexical). The
sparse side needs no corpus state client-side — fastembed computes the
stateless term-frequency half and the server's IDF modifier supplies document
frequencies live from the collection itself.
"""
import threading

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from ingestlib.config import get_qdrant_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: QdrantClient | None = None
_collection_dimension: int | None = None    # verified dense dimension, once ready
_bm25: SparseTextEmbedding | None = None

# Named vectors on every point.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# fastembed's BM25 encoder — tokenizes/stems locally, IDF stays server-side.
_BM25_MODEL = "Qdrant/bm25"

# Payload fields the store filters on. The real server rejects filters on
# unindexed fields (400) — local/in-memory mode does not enforce this, so
# these indexes are the difference between "works in dev" and "works in prod".
_INDEXED_FIELDS = ("namespace", "document_id", "category", "section", "kind")


def get_qdrant_client() -> QdrantClient:
    """Return the process-wide singleton Qdrant client."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_qdrant_config()
            logger.info("building Qdrant client: url=%s", cfg.url)
            _client = QdrantClient(url=cfg.url, api_key=cfg.api_key or None)
        return _client


def get_bm25() -> SparseTextEmbedding:
    """Return the process-wide BM25 encoder (small model files, cached on disk)."""
    global _bm25
    with _lock:
        if _bm25 is None:
            logger.info("loading BM25 sparse encoder (%s)", _BM25_MODEL)
            _bm25 = SparseTextEmbedding(_BM25_MODEL)
        return _bm25


def ensure_collection(dimension: int) -> str:
    """Create the configured collection on first use; no-op once it exists.

    Named vectors: "dense" (cosine; dimension from the first embedding batch,
    so collection shape always matches what is actually being stored) and
    "sparse" (BM25 with the server-side IDF modifier). Keyword payload
    indexes are (re)ensured for every filterable field. Returns the
    collection name.
    """
    global _collection_dimension
    cfg = get_qdrant_config()
    if _collection_dimension is not None:
        # the ready short-circuit must not skip the mismatch guard
        if _collection_dimension != dimension:
            raise ValueError(
                f"Qdrant collection {cfg.collection_name!r} has dense dimension "
                f"{_collection_dimension}, but embeddings have dimension {dimension} — "
                f"use a matching embedding dimension or a different collection"
            )
        return cfg.collection_name

    client = get_qdrant_client()
    if not client.collection_exists(cfg.collection_name):
        logger.info(
            "creating Qdrant collection %r (dense dim=%d cosine + sparse BM25) — first use",
            cfg.collection_name, dimension,
        )
        try:
            client.create_collection(
                collection_name=cfg.collection_name,
                vectors_config={
                    DENSE_VECTOR: VectorParams(size=dimension, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF),
                },
            )
        except Exception:
            # lost a creation race with another process — fine if it exists now
            if not client.collection_exists(cfg.collection_name):
                raise
    else:
        params = client.get_collection(cfg.collection_name).config.params
        vectors = params.vectors if isinstance(params.vectors, dict) else {}
        dense = vectors.get(DENSE_VECTOR)
        if dense is None:
            raise ValueError(
                f"Qdrant collection {cfg.collection_name!r} predates the named-vector "
                f"hybrid schema — delete it (or configure a different collection_name) "
                f"and re-ingest"
            )
        if dense.size != dimension:
            raise ValueError(
                f"Qdrant collection {cfg.collection_name!r} has dense dimension "
                f"{dense.size}, but embeddings have dimension {dimension} — "
                f"use a matching embedding dimension or a different collection"
            )

    # idempotent — also heals collections created before a field was added
    for field in _INDEXED_FIELDS:
        client.create_payload_index(
            collection_name=cfg.collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    _collection_dimension = dimension
    return cfg.collection_name


def reset_qdrant_client() -> None:
    """Force client recreation on the next call (e.g. after endpoint change)."""
    global _client, _collection_dimension
    with _lock:
        _client = None
        _collection_dimension = None
