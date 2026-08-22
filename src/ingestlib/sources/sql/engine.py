"""Read-only SQLAlchemy engine — connect, execute one query, return rows.

Engines (connection pools) are cached per DSN and disposed by reset_config().
The DSN carries the SQLAlchemy dialect (postgresql://…, mysql+pymysql://…,
sqlite:///…, duckdb:///…, snowflake://…), so this layer is dialect-agnostic
except for the optional per-query timeout it applies from dialects.py.
"""
import threading
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ingestlib.sources.sql.dialects import get_dialect
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_engines: dict[str, Engine] = {}


def get_engine(dsn: str) -> Engine:
    """A cached SQLAlchemy Engine for this DSN (pool_pre_ping guards stale conns)."""
    if not dsn or dsn.startswith("${"):
        raise RuntimeError(
            "SQL source DSN is unset — its ${VAR} did not resolve; set the "
            "connection URL in .env (a READ-ONLY role)"
        )
    with _lock:
        engine = _engines.get(dsn)
        if engine is None:
            logger.info("building SQL engine for %s", dsn.split("@")[-1])  # host only, no creds
            engine = create_engine(dsn, pool_pre_ping=True)
            _engines[dsn] = engine
        return engine


def reset_engines() -> None:
    """Dispose every cached engine (its pool) so the next call reconnects."""
    with _lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


def run_query(
    dsn: str,
    type_: str,
    sql: str,
    params: dict[str, Any],
    *,
    row_limit: int,
    timeout: int,
) -> tuple[list[str], list[tuple]]:
    """Execute `sql` (parameterized) read-only, returning (column names, rows).

    A dialect timeout is applied best-effort; `row_limit` caps rows via
    fetchmany regardless. Runs synchronously — callers use asyncio.to_thread.
    """
    engine = get_engine(dsn)
    dialect = get_dialect(type_)
    with engine.connect() as conn:
        if dialect.timeout_sql:
            try:
                conn.exec_driver_sql(
                    dialect.timeout_sql.format(sec=timeout, ms=timeout * 1000)
                )
            except Exception as exc:  # best-effort — row_limit is the hard bound
                logger.debug("could not apply %s timeout: %s", type_, exc)
        result = conn.execute(text(sql), params or {})
        columns = list(result.keys())
        rows = [tuple(r) for r in result.fetchmany(row_limit if row_limit > 0 else 1000)]
    return columns, rows
