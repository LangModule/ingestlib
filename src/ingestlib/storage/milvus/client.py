"""Process-wide singleton Milvus client + first-use collection bootstrap.

Works against a local standalone server (docker, embedded etcd) or Zilliz
Cloud (MILVUS_URL + MILVUS_TOKEN in .env; token empty for unsecured local).

One collection holds both retrieval signals:
    embedding — dense FLOAT_VECTOR (cosine, dimension from the first batch)
    sparse    — SPARSE_FLOAT_VECTOR the server computes ITSELF via the
                built-in BM25 function over `search_text` (Tantivy analyzer;
                IDF lives server-side — no corpus state, and callers insert
                and query with raw text)

Filter fields (namespace, document_id, category, section, kind) are scalar
columns usable in boolean filter expressions on every search. The collection
is created with Strong consistency so reads always see prior writes — no
eventual-consistency sleeps; at ingestlib's scale the latency cost is noise.
"""
import threading

from pymilvus import DataType, Function, FunctionType, MilvusClient

from ingestlib.config import get_milvus_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: MilvusClient | None = None
_ready: dict[tuple[str, str], int] = {}     # (url, collection) → dense dimension

# search_text carries breadcrumb + body; 65535 is VARCHAR's ceiling.
_TEXT_MAX = 65535


def get_milvus_client() -> MilvusClient:
    """Return the process-wide singleton Milvus client."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_milvus_config()
            logger.info("building Milvus client: url=%s", cfg.url)
            _client = MilvusClient(uri=cfg.url, token=cfg.token or "")
        return _client


def _build_schema(dimension: int):
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
    schema.add_field("document_id", DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_field("namespace", DataType.VARCHAR, max_length=256)
    schema.add_field("category", DataType.VARCHAR, max_length=256)
    schema.add_field("section", DataType.VARCHAR, max_length=512)
    schema.add_field("kind", DataType.VARCHAR, max_length=32)
    schema.add_field("search_text", DataType.VARCHAR, max_length=_TEXT_MAX,
                     enable_analyzer=True)
    schema.add_field("payload", DataType.JSON)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="bm25", function_type=FunctionType.BM25,
        input_field_names=["search_text"], output_field_names=["sparse"],
    ))
    return schema


def ensure_collection(dimension: int) -> str:
    """Create the configured collection on first use; verify it afterwards.

    The dense dimension comes from the first embedding batch, so the schema
    always matches what is actually being stored — later calls with a
    different dimension fail loudly. Returns the collection name.
    """
    cfg = get_milvus_config()
    key = (cfg.url, cfg.collection_name)
    client = get_milvus_client()
    with _lock:
        known = _ready.get(key)
        if known is not None:
            if known != dimension:
                raise ValueError(
                    f"Milvus collection {cfg.collection_name!r} stores {known}-dim "
                    f"embeddings, got {dimension}-dim — use a matching embedding "
                    f"dimension or a different collection_name"
                )
            return cfg.collection_name

        if not client.has_collection(cfg.collection_name):
            logger.info(
                "creating Milvus collection %r (dense dim=%d cosine + server-side"
                " BM25 sparse) — first use",
                cfg.collection_name, dimension,
            )
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(field_name="embedding", index_type="AUTOINDEX",
                                   metric_type="COSINE")
            index_params.add_index(field_name="sparse",
                                   index_type="SPARSE_INVERTED_INDEX",
                                   metric_type="BM25")
            client.create_collection(
                cfg.collection_name,
                schema=_build_schema(dimension),
                index_params=index_params,
                consistency_level="Strong",
            )
        else:
            desc = client.describe_collection(cfg.collection_name)
            fields = {f["name"]: f for f in desc["fields"]}
            if "embedding" not in fields or "sparse" not in fields:
                raise ValueError(
                    f"Milvus collection {cfg.collection_name!r} predates the hybrid "
                    f"schema — delete it (or configure a different collection_name) "
                    f"and re-ingest"
                )
            existing = int(fields["embedding"]["params"]["dim"])
            if existing != dimension:
                raise ValueError(
                    f"Milvus collection {cfg.collection_name!r} stores {existing}-dim "
                    f"embeddings, got {dimension}-dim — use a matching embedding "
                    f"dimension or a different collection_name"
                )

        _ready[key] = dimension
        return cfg.collection_name


def reset_milvus_client() -> None:
    """Force client recreation on the next call (e.g. after endpoint change)."""
    global _client
    with _lock:
        _client = None
        _ready.clear()
