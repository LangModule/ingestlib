"""get_dialect() — the per-type timeout mapping. Pure, always run."""
import pytest

from ingestlib.sources.sql.dialects import get_dialect


def test_known_types_resolve():
    for t in ("postgres", "mysql", "sqlite", "duckdb", "snowflake"):
        assert get_dialect(t).name == t


def test_server_timeout_sql_formats_with_sec_and_ms():
    assert (get_dialect("postgres").timeout_sql.format(sec=30, ms=30000)
            == "SET statement_timeout = 30000")
    assert (get_dialect("mysql").timeout_sql.format(sec=30, ms=30000)
            == "SET SESSION max_execution_time = 30000")
    # snowflake times out in whole seconds
    assert (get_dialect("snowflake").timeout_sql.format(sec=30, ms=30000)
            == "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 30")


def test_serverless_dialects_have_no_timeout_sql():
    assert get_dialect("sqlite").timeout_sql == ""
    assert get_dialect("duckdb").timeout_sql == ""


def test_unknown_type_raises_listing_the_valid_ones():
    with pytest.raises(ValueError, match="unknown SQL source type 'oracle'"):
        get_dialect("oracle")
