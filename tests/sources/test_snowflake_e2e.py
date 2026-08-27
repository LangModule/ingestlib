"""Snowflake structured-retrieval e2e — a real LLM over the built-in TPC-H sample
data, executed through a read-only role.

Opt-in via RUN_SNOWFLAKE_E2E=1; needs SNOWFLAKE_DSN in .env (a read-only role
with access to SNOWFLAKE_SAMPLE_DATA) and the configured LLM reachable. Uses the
pre-existing sample tables — a read-only role cannot create its own — so there is
no seeding or teardown. The DSN is read straight from .env (no sources.yaml
needed): config load populates os.environ, and we build the SourceSpec here."""
import os

import pytest

pytest.importorskip("snowflake.sqlalchemy")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SNOWFLAKE_E2E") != "1",
    reason="snowflake e2e is opt-in: set RUN_SNOWFLAKE_E2E=1 (needs SNOWFLAKE_DSN + an LLM)",
)

_REGION = "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION"


def _spec(dsn: str, **overrides):
    from ingestlib.config import SourceSpec

    fields = dict(
        name="tpch", type="snowflake", dsn=dsn, row_limit=50,
        description="TPC-H sample data: regions, nations, customers, orders",
        tables={
            "REGION": "5 rows: R_REGIONKEY, R_NAME (AFRICA, AMERICA, ASIA, EUROPE, MIDDLE EAST)",
            "NATION": "25 rows: N_NATIONKEY, N_NAME, N_REGIONKEY references REGION",
        },
    )
    fields.update(overrides)
    return SourceSpec(**fields)


@pytest.fixture()
def dsn():
    from ingestlib.config import get_config
    from ingestlib.sources.sql.engine import reset_engines

    get_config()  # loads .env → SNOWFLAKE_DSN into os.environ
    value = os.environ.get("SNOWFLAKE_DSN", "")
    if not value or value.startswith("${") or "<" in value:
        pytest.skip("SNOWFLAKE_DSN not set (or still a placeholder) in .env")
    reset_engines()
    yield value
    reset_engines()


async def test_health_reaches_snowflake(dsn):
    from ingestlib.sources.sql.source import SqlSource

    status, detail = await SqlSource(_spec(dsn)).health()
    assert status == "ok", detail


async def test_verified_query_returns_the_exact_region_count(dsn):
    from ingestlib.sources.sql.source import SqlSource

    verified = {"region_count": {
        "description": "how many regions are there",
        "sql": f"SELECT COUNT(*) AS n FROM {_REGION}",
    }}
    [r] = await SqlSource(_spec(dsn, verified=verified)).answer("how many regions are there?")
    assert r.provenance["verified"] is True
    assert r.raw["rows"] == [(5,)]


async def test_generated_sql_answers_a_schema_question(dsn):
    from ingestlib.sources.sql.source import SqlSource

    # no verified queries on this spec → the LLM must generate SQL from the hints
    [r] = await SqlSource(_spec(dsn)).answer("list the distinct region names")
    assert r.provenance["verified"] is False
    assert r.raw["rows"], "expected region rows from generated SQL"
    names = {str(v).upper() for row in r.raw["rows"] for v in row}
    assert names & {"ASIA", "EUROPE", "AMERICA", "AFRICA", "MIDDLE EAST"}


async def test_schema_rag_forced_on_still_answers(dsn):
    from ingestlib.sources.sql.source import SqlSource

    # force retrieval (schema_rag=on) on the clean TPC-H schema — introspection +
    # FK reflection + embedding retrieval must all work on Snowflake without
    # regressing the answer. min_tables=0 would also trigger it under auto.
    src = SqlSource(_spec(dsn, schema_rag="on", schema_rag_top_k=4))
    [r] = await src.answer("list the distinct region names")
    assert r.provenance["verified"] is False
    names = {str(v).upper() for row in r.raw["rows"] for v in row}
    assert names & {"ASIA", "EUROPE", "AMERICA", "AFRICA", "MIDDLE EAST"}
