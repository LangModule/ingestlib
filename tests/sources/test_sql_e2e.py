"""Structured-retrieval e2e with a REAL LLM generating SQL over local databases.

Opt-in via RUN_SQL_E2E=1 (needs the configured LLM reachable). SQLite and DuckDB
are serverless — seeded in a tmp dir, no container. Postgres runs too when
PGVECTOR_URL is set (reuse the pgvector container); MySQL when SQL_MYSQL_DSN is
set (mysql+pymysql://root:pw@localhost:3306/ingestlib — the compose `mysql`
profile). The model must translate the natural-language question into SQL that
finds the 2 'ready' rows — this is the real generate → guard → execute → render
loop, not a stub."""
import os

import pytest

pytest.importorskip("sqlalchemy")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SQL_E2E") != "1",
    reason="sql e2e is opt-in: set RUN_SQL_E2E=1 (needs the configured LLM)",
)

_ROWS = [(1, "ready", "2026-01-01"), (2, "ready", "2026-01-02"), (3, "pending", None)]


def _seed(dsn: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS rx"))
        conn.execute(text(
            "CREATE TABLE rx (rx_id INTEGER, status VARCHAR(16), ready_at VARCHAR(32))"
        ))
        for rx_id, status, ready_at in _ROWS:
            conn.execute(
                text("INSERT INTO rx (rx_id, status, ready_at) VALUES (:i, :s, :r)"),
                {"i": rx_id, "s": status, "r": ready_at},
            )
    engine.dispose()


def _drop(dsn: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS rx"))
    engine.dispose()


def _spec(name: str, type_: str, dsn: str):
    from ingestlib.config import SourceSpec

    return SourceSpec(
        name=name, type=type_, dsn=dsn, row_limit=100,
        description="prescription fills",
        tables={"rx": "one row per prescription; status is 'ready' or 'pending'"},
    )


async def _assert_finds_two_ready(source):
    [r] = await source.answer("how many prescriptions are ready?")
    assert r.provenance["verified"] is False
    flat = {str(v) for row in r.raw["rows"] for v in row}
    assert "2" in flat, f"expected the count 2 in generated result {r.raw['rows']}"


async def test_sqlite_generation(tmp_path):
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.sources.sql.source import SqlSource

    dsn = f"sqlite:///{tmp_path / 'rx.db'}"
    _seed(dsn)
    reset_engines()
    try:
        await _assert_finds_two_ready(SqlSource(_spec("rx", "sqlite", dsn)))
    finally:
        reset_engines()


async def test_duckdb_generation(tmp_path):
    pytest.importorskip("duckdb_engine")
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.sources.sql.source import SqlSource

    dsn = f"duckdb:///{tmp_path / 'rx.duckdb'}"
    _seed(dsn)
    reset_engines()
    try:
        await _assert_finds_two_ready(SqlSource(_spec("rx", "duckdb", dsn)))
    finally:
        reset_engines()


@pytest.mark.skipif(
    os.environ.get("RUN_PGVECTOR_E2E") != "1" or not os.environ.get("PGVECTOR_URL"),
    reason="postgres path is opt-in: RUN_PGVECTOR_E2E=1 + PGVECTOR_URL "
           "(reuse the pgvector container)",
)
async def test_postgres_generation():
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.sources.sql.source import SqlSource

    # the pgvector connector uses psycopg v3; point SQLAlchemy at the same driver
    dsn = os.environ["PGVECTOR_URL"].replace("postgresql://", "postgresql+psycopg://", 1)
    _seed(dsn)
    reset_engines()
    try:
        await _assert_finds_two_ready(SqlSource(_spec("rx", "postgres", dsn)))
    finally:
        _drop(dsn)
        reset_engines()


@pytest.mark.skipif(
    not os.environ.get("SQL_MYSQL_DSN"),
    reason="set SQL_MYSQL_DSN=mysql+pymysql://root:pw@localhost:3306/ingestlib (compose `mysql`)",
)
async def test_mysql_generation():
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.sources.sql.source import SqlSource

    dsn = os.environ["SQL_MYSQL_DSN"]
    _seed(dsn)
    reset_engines()
    try:
        await _assert_finds_two_ready(SqlSource(_spec("rx", "mysql", dsn)))
    finally:
        _drop(dsn)
        reset_engines()
