"""Process-wide singleton Pinecone client + first-time index bootstrap."""
import threading
import time

from pinecone import Pinecone, ServerlessSpec

from ingestlib.config import get_pinecone_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: Pinecone | None = None
_index_ready = False

# create_index is async server-side; poll readiness up to this long.
_READY_TIMEOUT_SECONDS = 60.0


def get_pinecone_client() -> Pinecone:
    """Return the process-wide singleton Pinecone client."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_pinecone_config()
            if not cfg.api_key:
                raise RuntimeError("PINECONE_API_KEY is not set — add it to .env")
            _client = Pinecone(api_key=cfg.api_key)
        return _client


def ensure_index(dimension: int) -> str:
    """Create the configured index on first use; no-op once it exists.

    Serverless index (cloud/region from config, cosine metric). The dimension
    comes from the first embedding batch, so index shape always matches what
    is actually being stored. Returns the index name.
    """
    global _index_ready
    cfg = get_pinecone_config()
    if _index_ready:
        return cfg.index_name

    client = get_pinecone_client()
    if not client.has_index(cfg.index_name):
        logger.info(
            "creating Pinecone serverless index %r (dim=%d, %s/%s) — first use",
            cfg.index_name, dimension, cfg.cloud, cfg.region,
        )
        try:
            client.create_index(
                name=cfg.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cfg.cloud, region=cfg.region),
            )
        except Exception:
            # lost a creation race with another process — fine if it exists now
            if not client.has_index(cfg.index_name):
                raise
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while not client.describe_index(cfg.index_name).status["ready"]:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Pinecone index {cfg.index_name!r} not ready after "
                    f"{_READY_TIMEOUT_SECONDS:.0f}s"
                )
            time.sleep(1.0)
        logger.info("Pinecone index %r ready", cfg.index_name)
    else:
        existing = client.describe_index(cfg.index_name)
        if existing.dimension != dimension:
            raise ValueError(
                f"Pinecone index {cfg.index_name!r} has dimension "
                f"{existing.dimension}, but embeddings have dimension {dimension} — "
                f"use a matching embedding dimension or a different index"
            )

    _index_ready = True
    return cfg.index_name


def reset_pinecone_client() -> None:
    """Force client recreation on the next call (e.g. after key rotation)."""
    global _client, _index_ready
    with _lock:
        _client = None
        _index_ready = False
