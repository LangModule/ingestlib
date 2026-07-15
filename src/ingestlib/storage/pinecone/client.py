"""Process-wide singleton Pinecone client + first-time index bootstrap.

Two indexes back the connector: the dense index (cosine, dimension from the
first embedding batch) and the sparse index (dotproduct, no dimension) that
holds the lexical half of hybrid search. Both create themselves on first use.

Sparse embeddings come from a Pinecone-hosted model (embed_sparse) — unlike
classic BM25 there is no corpus statistics state to fit, persist, or refresh.
"""
import threading
import time

from pinecone import Pinecone, ServerlessSpec

from ingestlib.config import get_pinecone_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: Pinecone | None = None
_index_dimension: int | None = None     # verified dense dimension, once ready
_sparse_index_ready = False

# create_index is async server-side; poll readiness up to this long.
_READY_TIMEOUT_SECONDS = 60.0

# The inference API caps inputs per embed call.
_SPARSE_EMBED_BATCH = 96


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


def _wait_until_ready(client: Pinecone, index_name: str) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while not client.describe_index(index_name).status["ready"]:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Pinecone index {index_name!r} not ready after "
                f"{_READY_TIMEOUT_SECONDS:.0f}s"
            )
        time.sleep(1.0)
    logger.info("Pinecone index %r ready", index_name)


def ensure_index(dimension: int) -> str:
    """Create the configured dense index on first use; no-op once it exists.

    Serverless index (cloud/region from config, cosine metric). The dimension
    comes from the first embedding batch, so index shape always matches what
    is actually being stored. Returns the index name.
    """
    global _index_dimension
    cfg = get_pinecone_config()
    if _index_dimension is not None:
        # the ready short-circuit must not skip the mismatch guard
        if _index_dimension != dimension:
            raise ValueError(
                f"Pinecone index {cfg.index_name!r} has dimension "
                f"{_index_dimension}, but embeddings have dimension {dimension} — "
                f"use a matching embedding dimension or a different index"
            )
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
        _wait_until_ready(client, cfg.index_name)
    else:
        existing = client.describe_index(cfg.index_name)
        if existing.dimension != dimension:
            raise ValueError(
                f"Pinecone index {cfg.index_name!r} has dimension "
                f"{existing.dimension}, but embeddings have dimension {dimension} — "
                f"use a matching embedding dimension or a different index"
            )

    _index_dimension = dimension
    return cfg.index_name


def ensure_sparse_index() -> str:
    """Create the configured sparse index on first use; no-op once it exists.

    Sparse serverless indexes take no dimension and require the dotproduct
    metric. Returns the index name.
    """
    global _sparse_index_ready
    cfg = get_pinecone_config()
    if _sparse_index_ready:
        return cfg.sparse_index_name

    client = get_pinecone_client()
    if not client.has_index(cfg.sparse_index_name):
        logger.info(
            "creating Pinecone sparse index %r (%s/%s) — first use",
            cfg.sparse_index_name, cfg.cloud, cfg.region,
        )
        try:
            client.create_index(
                name=cfg.sparse_index_name,
                metric="dotproduct",
                vector_type="sparse",
                spec=ServerlessSpec(cloud=cfg.cloud, region=cfg.region),
            )
        except Exception:
            # lost a creation race with another process — fine if it exists now
            if not client.has_index(cfg.sparse_index_name):
                raise
        _wait_until_ready(client, cfg.sparse_index_name)
    else:
        existing = client.describe_index(cfg.sparse_index_name)
        if existing.vector_type != "sparse":
            raise ValueError(
                f"Pinecone index {cfg.sparse_index_name!r} exists but is not a "
                f"sparse index (vector_type={existing.vector_type!r}) — "
                f"configure a different sparse_index_name"
            )

    _sparse_index_ready = True
    return cfg.sparse_index_name


def embed_sparse(texts: list[str], input_type: str) -> list[tuple[list[int], list[float]]]:
    """Sparse-embed texts via the hosted model → (indices, values) per text.

    input_type is "passage" for stored chunks or "query" for search queries —
    the model weighs tokens differently for each side.
    """
    client = get_pinecone_client()
    cfg = get_pinecone_config()
    out: list[tuple[list[int], list[float]]] = []
    for i in range(0, len(texts), _SPARSE_EMBED_BATCH):
        response = client.inference.embed(
            model=cfg.sparse_model_id,
            inputs=texts[i : i + _SPARSE_EMBED_BATCH],
            parameters={"input_type": input_type, "truncate": "END"},
        )
        out.extend((e.sparse_indices, e.sparse_values) for e in response.data)
    return out


def reset_pinecone_client() -> None:
    """Force client recreation on the next call (e.g. after key rotation)."""
    global _client, _index_dimension, _sparse_index_ready
    with _lock:
        _client = None
        _index_dimension = None
        _sparse_index_ready = False
