"""MongoDB connector — Atlas Vector Search + Atlas Search (true BM25) hybrid.

Works against Atlas (any tier), the mongodb-atlas-local docker image, or
self-managed MongoDB 8.2+ with mongot. One connection URL (MONGODB_URL in
.env) — both search indexes bootstrap on first use.
"""
from ingestlib.storage.mongodb.store import MongodbStore

__all__ = ["MongodbStore"]
