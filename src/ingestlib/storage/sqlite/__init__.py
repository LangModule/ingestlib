"""SQLite connector — zero-infrastructure vector storage in one local file.

Dense KNN via the sqlite-vec extension, lexical BM25 via built-in FTS5,
fused with RRF. No server, no credentials — sqlite.path in config.yaml.
"""
from ingestlib.storage.sqlite.store import SqliteStore

__all__ = ["SqliteStore"]
