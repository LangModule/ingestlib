"""Text2SQL quality eval — measures, never asserts.

Runs every question in sql_dataset.yaml through the real SqlSource.answer() path
(schema-introspected generation, verified-query matching, guardrails, self-correct)
against Snowflake's built-in TPC-H sample data, scores execution-match against
known answers, and prints:

    - overall match rate,
    - the match rate on GENERATED-only questions (the honest production number —
      how often the LLM writes correct SQL when there's no reviewed query to fall
      back on), and
    - the verified-vs-generated path split.

A timestamped snapshot lands in evals/results/. Like the retrieval eval, this is
a measurement harness — text2SQL accuracy drifts with the model and the schema
hints, so a report informs rather than a red CI run blocking.

Usage:
    RUN_SNOWFLAKE_E2E=1 uv run python evals/run_sql_eval.py     # needs SNOWFLAKE_DSN + an LLM
    uv run python evals/run_sql_eval.py --row-limit 100
"""
import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ingestlib.cli.eval_sql import print_report, run, summarize
from ingestlib.config import SourceSpec, get_config
from ingestlib.sources.sql.engine import reset_engines
from ingestlib.sources.sql.source import SqlSource

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

# Schema hints (the accuracy lever) + a couple of verified queries, so the report
# shows the verified path firing alongside pure generation. Everything else the
# model must generate from the introspected schema + these hints.
_TABLE_HINTS = {
    "REGION": "5 rows. R_REGIONKEY, R_NAME (AFRICA/AMERICA/ASIA/EUROPE/MIDDLE EAST), R_COMMENT",
    "NATION": "25 rows. N_NATIONKEY, N_NAME, N_REGIONKEY references REGION, N_COMMENT",
    "CUSTOMER": "C_CUSTKEY, C_NAME, C_NATIONKEY references NATION, C_ACCTBAL, C_MKTSEGMENT",
    "SUPPLIER": "S_SUPPKEY, S_NAME, S_NATIONKEY references NATION, S_ACCTBAL",
    "PART": "P_PARTKEY, P_NAME, P_MFGR, P_BRAND, P_TYPE, P_RETAILPRICE",
    "ORDERS": "O_ORDERKEY, O_CUSTKEY references CUSTOMER, O_ORDERSTATUS, O_ORDERPRIORITY",
    "LINEITEM": "one row per order line. L_ORDERKEY, L_PARTKEY, L_QUANTITY, L_EXTENDEDPRICE",
    "PARTSUPP": "PS_PARTKEY, PS_SUPPKEY, PS_AVAILQTY, PS_SUPPLYCOST",
}

_VERIFIED = {
    "region_count": {
        "description": "how many regions are there",
        "sql": "SELECT COUNT(*) AS n FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.REGION",
    },
    "nation_count": {
        "description": "how many nations are there in total",
        "sql": "SELECT COUNT(*) AS n FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION",
    },
}


# dataset `source:` names → the spec builder for that database. One entry today;
# add a builder here (and a case below) to eval a second source.
_SUPPORTED_SOURCES = {"snowflake_tpch"}


def build_source(name: str, row_limit: int) -> SqlSource:
    """Build the SqlSource for a dataset `source:` name. Only snowflake_tpch today
    (built from SNOWFLAKE_DSN in .env); add an `elif` here to support another."""
    get_config()  # loads .env → connection URLs into os.environ
    if name == "snowflake_tpch":
        dsn = os.environ.get("SNOWFLAKE_DSN", "")
        if not dsn or dsn.startswith("${") or "<" in dsn:
            raise SystemExit(
                "SNOWFLAKE_DSN is not set (or still a placeholder) in .env — this eval "
                "runs against the TPC-H sample data via a read-only Snowflake role."
            )
        return SqlSource(SourceSpec(
            name=name, type="snowflake", dsn=dsn, row_limit=row_limit,
            description="TPC-H sample data: regions, nations, customers, orders, parts",
            tables=_TABLE_HINTS, verified=_VERIFIED,
        ))
    raise SystemExit(f"no builder for source {name!r}")  # unreachable via load_dataset's guard


def load_dataset() -> tuple[str, list[dict]]:
    """Return (source name, questions). The `source:` key selects which database
    the questions run against (validated against the supported builders)."""
    data = yaml.safe_load((EVALS_DIR / "sql_dataset.yaml").read_text())
    source = data.get("source", "")
    if source not in _SUPPORTED_SOURCES:
        raise SystemExit(
            f"sql_dataset.yaml `source: {source!r}` is not supported — one of "
            f"{sorted(_SUPPORTED_SOURCES)} (each maps to a builder in build_source)"
        )
    questions = data["questions"]
    ids = [q["id"] for q in questions]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"sql_dataset.yaml has duplicate ids: {dupes}")
    return source, questions


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-limit", type=int, default=100)
    args = parser.parse_args()

    source_name, dataset = load_dataset()
    source = build_source(source_name, args.row_limit)
    print(f"running {len(dataset)} text2SQL questions against {source_name} ...")
    reset_engines()
    rows = await run(source, dataset)
    summary = summarize(rows)
    print_report(summary, rows)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"sql-eval-{source_name}-{stamp}.json"
    out.write_text(json.dumps({
        "source": source_name,
        "llm_provider": get_config().llm_provider,
        "embedding_provider": get_config().embedding_provider,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "questions": rows,
    }, indent=2))
    print(f"\nsaved {out.relative_to(EVALS_DIR.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
