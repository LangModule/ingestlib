"""Postgres connection factory + first-use extension and schema bootstrap.

The bring-your-own-Postgres connector: the user provides ONE connection URL
(PGVECTOR_URL in .env) and everything else bootstraps — the vector extension
is verified and enabled if the server ships it, and the table/indexes create
themselves on first use, like every other connector's bucket/index/collection.

One table holds everything per chunk:
    embedding   — pgvector column, HNSW cosine index (iterative scans on,
                  so namespace/metadata filters don't starve the KNN)
    fts         — weighted tsvector GENERATED from breadcrumb + body: the
                  lexical index cannot drift from the row store, by the
                  type system (stronger than sqlite's trigger sync)
    payload     — full chunk provenance as JSONB
    filter cols — namespace, document_id, category, section, kind (btree)

Connections are short-lived (one per store operation), autocommit with
explicit transaction blocks for multi-statement writes.
"""
import re
import threading

import psycopg
from pgvector.psycopg import register_vector

from ingestlib.config import get_pgvector_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_extension_ready: set[str] = set()          # URLs whose server has the extension enabled
_ready: dict[tuple[str, str], int] = {}     # (url, table) → dense dimension, once verified

# table_name is interpolated into SQL — it must be a plain identifier.
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def table_name() -> str:
    """The configured table name, validated as a safe SQL identifier."""
    name = get_pgvector_config().table_name
    if not _IDENTIFIER.match(name):
        raise ValueError(
            f"pgvector table_name {name!r} must be a plain lowercase identifier "
            f"(letters, digits, underscores)"
        )
    return name


def _ensure_extension(conn: psycopg.Connection, url: str) -> None:
    """Verify the vector extension: use it if installed, install it if the
    server ships it, and fail with the exact remedy otherwise."""
    if url in _extension_ready:
        return
    if conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone():
        _extension_ready.add(url)
        return
    available = conn.execute(
        "SELECT default_version FROM pg_available_extensions WHERE name = 'vector'"
    ).fetchone()
    if available is None:
        raise RuntimeError(
            "this Postgres server does not ship the pgvector extension — "
            "install it on the server (https://github.com/pgvector/pgvector) "
            "or use a provider that includes it (RDS, Supabase, Neon, "
            "or the pgvector/pgvector docker image)"
        )
    try:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.errors.InsufficientPrivilege as exc:
        raise RuntimeError(
            "pgvector is available on this server but not enabled, and this "
            "connection lacks the privilege to enable it — run "
            "'CREATE EXTENSION vector;' once as a superuser (or your "
            "provider's dashboard equivalent)"
        ) from exc
    logger.info("enabled pgvector extension %s — first use", available[0])
    _extension_ready.add(url)


def connect() -> psycopg.Connection:
    """One connection for one store operation: extension verified, vector
    type registered, iterative index scans enabled."""
    cfg = get_pgvector_config()
    if not cfg.url:
        raise RuntimeError(
            "PGVECTOR_URL is not set — add it to .env "
            "(postgresql://user:password@host:port/database)"
        )
    conn = psycopg.connect(cfg.url, autocommit=True)
    try:
        _ensure_extension(conn, cfg.url)
        register_vector(conn)
        try:
            # pre-0.8 servers lack the GUC — filtered KNN still works, just
            # with the old non-iterative behavior
            conn.execute("SET hnsw.iterative_scan = 'relaxed_order'")
        except psycopg.Error:
            pass
    except Exception:
        conn.close()
        raise
    return conn


def schema_exists(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (table_name(),)).fetchone()
    return row is not None and row[0] is not None


def _existing_dimension(conn: psycopg.Connection, table: str) -> int:
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = %s::regclass AND attname = 'embedding'",
        (table,),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"Postgres table {table!r} already exists but has no 'embedding' "
            f"column — it is not ingestlib's; configure a different table_name"
        )
    return int(row[0])


def ensure_schema(conn: psycopg.Connection, dimension: int) -> None:
    """Create the table + indexes on first use; verify them afterwards.

    The dense dimension comes from the first embedding batch, so the vector
    column always matches what is actually being stored — later calls with a
    different dimension fail loudly instead of storing garbage distances.
    """
    cfg = get_pgvector_config()
    table = table_name()
    key = (cfg.url, table)
    with _lock:
        known = _ready.get(key)
        if known is not None:
            if known != dimension:
                raise ValueError(
                    f"Postgres table {table!r} stores {known}-dim "
                    f"embeddings, got {dimension}-dim — use a matching embedding "
                    f"dimension or a different table_name"
                )
            return

        if not schema_exists(conn):
            logger.info(
                "creating Postgres table %r (dense dim=%d cosine + weighted tsvector)"
                " — first use",
                table, dimension,
            )
            with conn.transaction():
                conn.execute(f"""
                    CREATE TABLE {table} (
                        id bigserial PRIMARY KEY,
                        document_id text NOT NULL,
                        chunk_id int NOT NULL,
                        namespace text NOT NULL DEFAULT '',
                        category text NOT NULL DEFAULT '',
                        section text NOT NULL DEFAULT '',
                        kind text NOT NULL DEFAULT 'text',
                        breadcrumb text NOT NULL DEFAULT '',
                        body text NOT NULL DEFAULT '',
                        payload jsonb NOT NULL,
                        embedding vector({dimension}),
                        fts tsvector GENERATED ALWAYS AS (
                            setweight(to_tsvector('english', breadcrumb), 'A') ||
                            setweight(to_tsvector('english', body), 'B')) STORED,
                        UNIQUE(document_id, chunk_id, namespace)
                    )
                """)
                conn.execute(
                    f"CREATE INDEX {table}_embedding_idx ON {table}"
                    f" USING hnsw (embedding vector_cosine_ops)"
                )
                conn.execute(
                    f"CREATE INDEX {table}_fts_idx ON {table} USING gin (fts)"
                )
                conn.execute(
                    f"CREATE INDEX {table}_document_idx ON {table}"
                    f" (namespace, document_id)"
                )
        else:
            existing = _existing_dimension(conn, table)
            if existing != dimension:
                raise ValueError(
                    f"Postgres table {table!r} stores {existing}-dim "
                    f"embeddings, got {dimension}-dim — use a matching embedding "
                    f"dimension or a different table_name"
                )

        _ready[key] = dimension


def reset_pgvector() -> None:
    """Forget bootstrap state so the next call re-verifies (e.g. URL change)."""
    with _lock:
        _extension_ready.clear()
        _ready.clear()
