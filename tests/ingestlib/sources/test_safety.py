"""guard() and with_limit() — the statement guardrails. Pure, always run."""
import pytest

from ingestlib.sources.sql.safety import UnsafeQuery, guard, with_limit


# ---- guard ----

def test_guard_accepts_a_plain_select():
    assert guard("SELECT * FROM rx", allow=("select",)) == "SELECT * FROM rx"


def test_guard_strips_trailing_semicolon_and_whitespace():
    assert guard("  SELECT 1 ;  ", allow=("select",)) == "SELECT 1"


def test_guard_is_case_insensitive_on_the_keyword():
    assert guard("select 1", allow=("select",)) == "select 1"
    assert guard("SeLeCt 1", allow=("select",)) == "SeLeCt 1"


def test_guard_rejects_empty():
    with pytest.raises(UnsafeQuery, match="empty"):
        guard("   ;  ", allow=("select",))


def test_guard_rejects_a_disallowed_keyword():
    with pytest.raises(UnsafeQuery, match="delete"):
        guard("DELETE FROM rx", allow=("select",))


def test_guard_names_the_allowed_set_in_the_error():
    with pytest.raises(UnsafeQuery, match="allowed: select"):
        guard("DROP TABLE rx", allow=("select",))


def test_guard_rejects_stacked_statements():
    with pytest.raises(UnsafeQuery, match="one query"):
        guard("SELECT 1; DROP TABLE rx", allow=("select",))


def test_guard_honors_a_wider_allowlist():
    sql = "WITH x AS (SELECT 1) SELECT * FROM x"
    assert guard(sql, allow=("with", "select")) == sql
    # ...but the same query is rejected when only select is allowed
    with pytest.raises(UnsafeQuery, match="with"):
        guard(sql, allow=("select",))


# ---- with_limit ----

def test_with_limit_injects_when_absent():
    assert with_limit("SELECT * FROM rx", 100).endswith("LIMIT 100")


def test_with_limit_leaves_an_existing_limit_alone():
    sql = "SELECT * FROM rx LIMIT 5"
    assert with_limit(sql, 100) == sql


def test_with_limit_is_case_insensitive_about_an_existing_limit():
    sql = "select * from rx limit 5"
    assert with_limit(sql, 100) == sql


def test_with_limit_is_a_noop_when_row_limit_not_positive():
    assert with_limit("SELECT 1", 0) == "SELECT 1"
    assert with_limit("SELECT 1", -1) == "SELECT 1"
