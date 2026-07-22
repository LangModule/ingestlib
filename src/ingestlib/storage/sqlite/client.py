"""SQLite connection factory + first-use schema bootstrap.

The zero-infrastructure connector: no server, no credentials — the database
is one local file (sqlite.path in config.yaml), created on first use. The
sqlite-vec extension supplies the vector side; extensions load per connection,
never "install" into the file, so every connection made here loads it.

Three tables share one rowid per chunk:
    chunks      — canonical row store: payload JSON + the text FTS indexes
    chunks_vec  — vec0 KNN table: embedding (cosine), namespace as PARTITION
                  KEY (physical shard), filter fields as indexed metadata
                  columns (pre-filtered KNN)
    chunks_fts  — FTS5 external-content index over chunks (porter stemming;
                  indexes chunks' text without storing a copy). Triggers keep
                  it in sync with chunks, so the lexical index cannot drift
                  from the row store. chunks_vec has no triggers — embeddings
                  arrive from outside, the store writes it explicitly.

Connections are short-lived (one per store operation); WAL mode makes opens
cheap and lets readers proceed during writes.
"""
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

from ingestlib.config import get_sqlite_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_ready: dict[Path, int] = {}    # db path → dense dimension, once the schema is verified

_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE chunks (
    rowid INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    namespace TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    section TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'text',
    breadcrumb TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    UNIQUE(document_id, chunk_id, namespace)
);

CREATE INDEX chunks_by_document ON chunks(namespace, document_id);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[{dimension}] distance_metric=cosine,
    namespace TEXT PARTITION KEY,
    document_id TEXT,
    category TEXT,
    section TEXT,
    kind TEXT
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    breadcrumb, body,
    content=chunks, content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, breadcrumb, body)
    VALUES (new.rowid, new.breadcrumb, new.body);
END;

CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, breadcrumb, body)
    VALUES ('delete', old.rowid, old.breadcrumb, old.body);
END;

CREATE TRIGGER chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, breadcrumb, body)
    VALUES ('delete', old.rowid, old.breadcrumb, old.body);
    INSERT INTO chunks_fts(rowid, breadcrumb, body)
    VALUES (new.rowid, new.breadcrumb, new.body);
END;
"""


def reset_sqlite() -> None:
    """Forget verified-schema state so the next call re-verifies (e.g. path change)."""
    with _lock:
        _ready.clear()


def _connect(path: Path) -> sqlite3.Connection:
    # isolation_level=None → autocommit; the store issues BEGIN/COMMIT itself
    conn = sqlite3.connect(path, isolation_level=None)
    if not hasattr(conn, "enable_load_extension"):
        conn.close()
        raise RuntimeError(
            "this Python was built without SQLite extension support, so the "
            "sqlite-vec extension cannot load (macOS's system Python is the "
            "usual culprit) — use a python.org, homebrew, or uv-managed "
            "interpreter"
        )
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    """One connection for one store operation, vec extension loaded."""
    cfg = get_sqlite_config()
    cfg.path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(cfg.path)
    try:
        yield conn
    finally:
        conn.close()


def schema_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()
    return row is not None


def _existing_dimension(conn: sqlite3.Connection) -> int:
    (sql,) = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks_vec'"
    ).fetchone()
    match = re.search(r"float\[(\d+)\]", sql)
    if match is None:
        raise ValueError(f"cannot read dense dimension from chunks_vec schema: {sql!r}")
    return int(match.group(1))


def ensure_schema(conn: sqlite3.Connection, dimension: int) -> None:
    """Create the three-table schema on first use; verify it afterwards.

    The dense dimension comes from the first embedding batch, so the vec table
    always matches what is actually being stored — later calls with a
    different dimension fail loudly instead of storing garbage distances.
    """
    cfg = get_sqlite_config()
    with _lock:
        known = _ready.get(cfg.path)
        if known is not None:
            if known != dimension:
                raise ValueError(
                    f"SQLite database {cfg.path} stores {known}-dim embeddings, "
                    f"got {dimension}-dim — use a matching embedding dimension "
                    f"or a different database file"
                )
            return

        if not schema_exists(conn):
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
            ).fetchone():
                raise ValueError(
                    f"SQLite database {cfg.path} already has a 'chunks' table that "
                    f"is not ingestlib's — point sqlite.path at a dedicated file"
                )
            logger.info(
                "creating SQLite schema in %s (dense dim=%d cosine + FTS5 BM25) — first use",
                cfg.path, dimension,
            )
            conn.execute("BEGIN IMMEDIATE")
            try:
                # re-check under the write lock — another PROCESS may have
                # created the schema between the check above and BEGIN
                if not schema_exists(conn):
                    # statements are separated by blank lines — the split
                    # depends on that formatting
                    for statement in _SCHEMA.format(dimension=dimension).split(";\n\n"):
                        conn.execute(statement)
                    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        else:
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            if version != _SCHEMA_VERSION:
                raise ValueError(
                    f"SQLite database {cfg.path} has schema version {version}, "
                    f"this ingestlib expects {_SCHEMA_VERSION} — re-ingest into a "
                    f"fresh database file"
                )
            existing = _existing_dimension(conn)
            if existing != dimension:
                raise ValueError(
                    f"SQLite database {cfg.path} stores {existing}-dim embeddings, "
                    f"got {dimension}-dim — use a matching embedding dimension "
                    f"or a different database file"
                )

        _ready[cfg.path] = dimension
