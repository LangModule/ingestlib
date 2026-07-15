"""Process-wide singleton MongoDB client + first-use search-index bootstrap.

Works against MongoDB Atlas (any tier — M0 free tier included), the
mongodb/mongodb-atlas-local docker image, or self-managed MongoDB 8.2+
running the mongot search binary. One connection URL (MONGODB_URL in .env).

One collection holds the chunks; two search indexes serve the two retrieval
signals, both created programmatically on first use:
    vector_index — Atlas Vector Search: cosine KNN over `embedding`
                   (dimension from the first batch), with every filterable
                   field declared as a filter path
    text_index   — Atlas Search: true BM25 over breadcrumb + body (lucene
                   analyzer stems english), filterable fields as token paths

Search-index builds are asynchronous server-side — creation polls until both
indexes are queryable, like Pinecone's readiness wait. The indexes also lag
writes by a few seconds (eventual consistency); callers that write-then-query
immediately must expect that.
"""
import threading
import time

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel

from ingestlib.config import get_mongodb_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: MongoClient | None = None
_ready: dict[tuple[str, str], int] = {}     # (database, collection) → dense dimension

VECTOR_INDEX = "vector_index"
TEXT_INDEX = "text_index"

# Fields declared filterable in BOTH indexes — must match the store's
# _FILTERABLE plus namespace.
_FILTER_FIELDS = ("namespace", "document_id", "category", "section", "kind")

# Index builds are asynchronous server-side; poll readiness up to this long.
_READY_TIMEOUT_SECONDS = 120.0


def get_mongodb_client() -> MongoClient:
    """Return the process-wide singleton MongoDB client (pooled, thread-safe)."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_mongodb_config()
            if not cfg.url:
                raise RuntimeError(
                    "MONGODB_URL is not set — add it to .env "
                    "(mongodb+srv://... from Atlas, or mongodb://host:port)"
                )
            logger.info("building MongoDB client: db=%s", cfg.database)
            _client = MongoClient(cfg.url)
        return _client


def get_collection() -> Collection:
    """The configured chunks collection."""
    cfg = get_mongodb_config()
    return get_mongodb_client()[cfg.database][cfg.collection_name]


def _vector_index_model(dimension: int) -> SearchIndexModel:
    return SearchIndexModel(
        name=VECTOR_INDEX,
        type="vectorSearch",
        definition={
            "fields": [
                {"type": "vector", "path": "embedding",
                 "numDimensions": dimension, "similarity": "cosine"},
                *({"type": "filter", "path": field} for field in _FILTER_FIELDS),
            ]
        },
    )


def _text_index_model() -> SearchIndexModel:
    return SearchIndexModel(
        name=TEXT_INDEX,
        type="search",
        definition={
            "mappings": {
                "dynamic": False,
                "fields": {
                    "breadcrumb": {"type": "string"},
                    "body": {"type": "string"},
                    **{field: {"type": "token"} for field in _FILTER_FIELDS},
                },
            }
        },
    )


def _wait_until_queryable(collection: Collection, names: set[str]) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while True:
        status = {
            idx["name"]: idx.get("queryable", False)
            for idx in collection.list_search_indexes()
            if idx["name"] in names
        }
        if len(status) == len(names) and all(status.values()):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"MongoDB search indexes not queryable after "
                f"{_READY_TIMEOUT_SECONDS:.0f}s: {status}"
            )
        time.sleep(2.0)


def ensure_collection(dimension: int) -> Collection:
    """Create both search indexes on first use; verify them afterwards.

    The dense dimension comes from the first embedding batch, so the vector
    index always matches what is actually being stored — later calls with a
    different dimension fail loudly instead of returning empty searches.
    """
    cfg = get_mongodb_config()
    key = (cfg.database, cfg.collection_name)
    collection = get_collection()
    with _lock:
        known = _ready.get(key)
        if known is not None:
            if known != dimension:
                raise ValueError(
                    f"MongoDB collection {cfg.collection_name!r} indexes "
                    f"{known}-dim embeddings, got {dimension}-dim — use a matching "
                    f"embedding dimension or a different collection_name"
                )
            return collection

        existing = {idx["name"]: idx for idx in collection.list_search_indexes()}
        missing = []
        if VECTOR_INDEX not in existing:
            missing.append(_vector_index_model(dimension))
        else:
            fields = existing[VECTOR_INDEX]["latestDefinition"]["fields"]
            indexed_dim = next(
                f["numDimensions"] for f in fields if f["type"] == "vector"
            )
            if indexed_dim != dimension:
                raise ValueError(
                    f"MongoDB collection {cfg.collection_name!r} indexes "
                    f"{indexed_dim}-dim embeddings, got {dimension}-dim — use a "
                    f"matching embedding dimension or a different collection_name"
                )
        if TEXT_INDEX not in existing:
            missing.append(_text_index_model())

        if missing:
            logger.info(
                "creating %d MongoDB search index(es) on %s.%s (dense dim=%d cosine"
                " + BM25 text) — first use; builds are async, polling readiness",
                len(missing), cfg.database, cfg.collection_name, dimension,
            )
            # search indexes need the collection to exist first
            if cfg.collection_name not in collection.database.list_collection_names():
                try:
                    collection.database.create_collection(cfg.collection_name)
                except CollectionInvalid:
                    pass  # another process won the creation race
            collection.create_search_indexes(missing)

    # poll OUTSIDE the lock — up to 120s; holding it would stall every other
    # mongo-touching thread behind an index build. A concurrent thread may
    # duplicate the wait, which is harmless.
    _wait_until_queryable(collection, {VECTOR_INDEX, TEXT_INDEX})

    with _lock:
        _ready[key] = dimension
    return collection


def reset_mongodb_client() -> None:
    """Force client recreation on the next call (e.g. after URL change)."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
        _client = None
        _ready.clear()
