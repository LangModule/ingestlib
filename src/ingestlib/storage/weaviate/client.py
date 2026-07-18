"""Process-wide singleton Weaviate client + first-use collection bootstrap.

Works against a local server (docker, both the HTTP and gRPC ports
published — the v4 client speaks gRPC for data operations) or Weaviate
Cloud (WEAVIATE_URL + WEAVIATE_API_KEY in .env).

One collection holds both retrieval signals: HNSW cosine over the
caller-supplied vector, and the server's native BM25 over the searchable
text properties (breadcrumb + body). Filter fields (namespace, document_id,
category, section, kind) are field-tokenized TEXT properties, so filters
match exact values, never word fragments. The payload rides as an unindexed
JSON string and is returned verbatim on hits.

Weaviate does not record a vector dimension in the schema for
self-provided vectors — it is fixed by the first stored vector — so the
dimension guard here is process-level; a mismatch against an existing
collection surfaces as a server error on insert or query.
"""
import threading
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.init import Auth

from ingestlib.config import get_weaviate_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: weaviate.WeaviateClient | None = None
_ready: dict[tuple[str, str], int] = {}     # (url, collection) → dense dimension

# The v4 client's data path is gRPC; local servers publish it here.
_GRPC_PORT = 50051

# Filter-only fields: exact-match tokenization, excluded from BM25.
_FILTER_FIELDS = ("namespace", "document_id", "category", "section", "kind")


def collection_name() -> str:
    """The configured collection name, capitalized — Weaviate requires it."""
    name = get_weaviate_config().collection_name
    return name[:1].upper() + name[1:]


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return the process-wide singleton Weaviate client."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_weaviate_config()
            parsed = urlparse(cfg.url)
            host = parsed.hostname or "localhost"
            logger.info("building Weaviate client: url=%s", cfg.url)
            if host.endswith(".weaviate.cloud"):
                _client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=cfg.url,
                    auth_credentials=Auth.api_key(cfg.api_key),
                )
            else:
                secure = parsed.scheme == "https"
                _client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=parsed.port or (443 if secure else 8080),
                    http_secure=secure,
                    grpc_host=host,
                    grpc_port=_GRPC_PORT,
                    grpc_secure=secure,
                    auth_credentials=Auth.api_key(cfg.api_key) if cfg.api_key else None,
                )
        return _client


def _properties() -> list[Property]:
    filter_props = [
        Property(
            name=field,
            data_type=DataType.TEXT,
            index_searchable=False,
            tokenization=Tokenization.FIELD,
        )
        for field in _FILTER_FIELDS
    ]
    return [
        *filter_props,
        Property(name="breadcrumb", data_type=DataType.TEXT),
        Property(name="body", data_type=DataType.TEXT),
        Property(
            name="payload",
            data_type=DataType.TEXT,
            index_searchable=False,
            index_filterable=False,
        ),
    ]


def ensure_collection(dimension: int) -> str:
    """Create the configured collection on first use; no-op once it exists.

    The process-level dimension guard rejects mixed dimensions within a run;
    against a pre-existing collection the server enforces it. Returns the
    collection name.
    """
    cfg = get_weaviate_config()
    name = collection_name()
    key = (cfg.url, name)
    client = get_weaviate_client()
    with _lock:
        known = _ready.get(key)
        if known is not None:
            if known != dimension:
                raise ValueError(
                    f"Weaviate collection {name!r} stores {known}-dim embeddings, "
                    f"got {dimension}-dim — use a matching embedding dimension or "
                    f"a different collection_name"
                )
            return name

        if not client.collections.exists(name):
            logger.info(
                "creating Weaviate collection %r (dense dim=%d cosine + BM25 text)"
                " — first use",
                name, dimension,
            )
            try:
                client.collections.create(
                    name=name,
                    vector_config=Configure.Vectors.self_provided(
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                        ),
                    ),
                    properties=_properties(),
                )
            except Exception:
                # lost a creation race with another process — fine if it exists now
                if not client.collections.exists(name):
                    raise

        _ready[key] = dimension
        return name


def reset_weaviate_client() -> None:
    """Force client recreation on the next call (e.g. after endpoint change)."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
        _client = None
        _ready.clear()
