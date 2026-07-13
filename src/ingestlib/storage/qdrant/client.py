"""Process-wide singleton Qdrant client + first-time collection bootstrap.

Works against a local server (docker run -p 6333:6333 qdrant/qdrant, no API
key) or Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY in .env).
"""
import threading

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ingestlib.config import get_qdrant_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: QdrantClient | None = None
_collection_ready = False


def get_qdrant_client() -> QdrantClient:
    """Return the process-wide singleton Qdrant client."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_qdrant_config()
            logger.info("building Qdrant client: url=%s", cfg.url)
            _client = QdrantClient(url=cfg.url, api_key=cfg.api_key or None)
        return _client


def ensure_collection(dimension: int) -> str:
    """Create the configured collection on first use; no-op once it exists.

    Cosine distance; the dimension comes from the first embedding batch, so
    collection shape always matches what is actually being stored. Returns
    the collection name.
    """
    global _collection_ready
    cfg = get_qdrant_config()
    if _collection_ready:
        return cfg.collection_name

    client = get_qdrant_client()
    if not client.collection_exists(cfg.collection_name):
        logger.info(
            "creating Qdrant collection %r (dim=%d, cosine) — first use",
            cfg.collection_name, dimension,
        )
        try:
            client.create_collection(
                collection_name=cfg.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
        except Exception:
            # lost a creation race with another process — fine if it exists now
            if not client.collection_exists(cfg.collection_name):
                raise
    else:
        existing = client.get_collection(cfg.collection_name).config.params.vectors.size
        if existing != dimension:
            raise ValueError(
                f"Qdrant collection {cfg.collection_name!r} has dimension "
                f"{existing}, but embeddings have dimension {dimension} — "
                f"use a matching embedding dimension or a different collection"
            )

    _collection_ready = True
    return cfg.collection_name


def reset_qdrant_client() -> None:
    """Force client recreation on the next call (e.g. after endpoint change)."""
    global _client, _collection_ready
    with _lock:
        _client = None
        _collection_ready = False
