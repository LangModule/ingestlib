"""Engine caching + run_query — against a real on-disk SQLite DB (serverless,
so ungated). The unresolved-DSN guards are pure."""
from ingestlib.sources.sql import engine
from ingestlib.sources.sql.engine import get_engine, reset_engines, run_query

import pytest


def test_unset_dsn_raises_a_read_only_hint():
    with pytest.raises(RuntimeError, match="READ-ONLY"):
        get_engine("")


def test_unresolved_var_dsn_raises():
    with pytest.raises(RuntimeError, match="did not resolve"):
        get_engine("${RX_DB_DSN}")


def test_engine_is_cached_per_dsn(rx_dsn):
    assert get_engine(rx_dsn) is get_engine(rx_dsn)


def test_reset_engines_disposes_and_clears(rx_dsn):
    get_engine(rx_dsn)
    assert engine._engines, "engine should be cached after get_engine"
    reset_engines()
    assert engine._engines == {}


def test_run_query_returns_columns_and_rows(rx_dsn):
    cols, rows = run_query(
        rx_dsn, "sqlite", "SELECT rx_id, status FROM rx ORDER BY rx_id", {},
        row_limit=1000, timeout=30,
    )
    assert cols == ["rx_id", "status"]
    assert rows[0] == (1, "ready")
    assert len(rows) == 3


def test_run_query_row_limit_caps_via_fetchmany(rx_dsn):
    _, rows = run_query(
        rx_dsn, "sqlite", "SELECT * FROM rx", {}, row_limit=2, timeout=30,
    )
    assert len(rows) == 2


def test_run_query_binds_named_params(rx_dsn):
    _, rows = run_query(
        rx_dsn, "sqlite", "SELECT rx_id FROM rx WHERE status = :st", {"st": "ready"},
        row_limit=1000, timeout=30,
    )
    assert {r[0] for r in rows} == {1, 2}
