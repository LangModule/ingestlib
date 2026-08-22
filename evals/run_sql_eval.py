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


def _cells(rows: list[tuple]) -> list[str]:
    """Every value in the result, stringified and stripped — the haystack."""
    return [str(v).strip() for row in rows for v in row if v is not None]


def matched(expect, rows: list[tuple]) -> bool:
    """A hit when any expected value is present: numbers match a cell exactly (so
    '5' never matches '25'); text matches a cell case-insensitively as substring."""
    wants = expect if isinstance(expect, list) else [expect]
    cells = _cells(rows)
    lowered = [c.lower() for c in cells]
    for want in wants:
        want = str(want).strip()
        if want.replace(",", "").isdigit():
            if want in cells or want.replace(",", "") in [c.replace(",", "") for c in cells]:
                return True
        elif any(want.lower() in c for c in lowered):
            return True
    return False


async def run(source: SqlSource, dataset: list[dict]) -> list[dict]:
    rows_out = []
    for q in dataset:
        entry = {"id": q["id"], "question": q["question"], "expect": q["expect"]}
        try:
            [result] = await source.answer(q["question"])
            entry["ran"] = True
            entry["path"] = "verified" if result.provenance.get("verified") else "generated"
            entry["sql"] = result.provenance.get("sql", "")
            entry["hit"] = matched(q["expect"], result.raw["rows"])
        except Exception as exc:  # a query that never ran (guard reject, DB error, ...)
            entry.update(ran=False, path="error", sql="", hit=False, error=str(exc)[:200])
        rows_out.append(entry)
        mark = "✓" if entry["hit"] else ("·" if entry["ran"] else "✗")
        print(f"  {mark} [{entry['path']:<9}] {q['id']}")
    return rows_out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    gen = [r for r in rows if r["path"] == "generated"]
    ver = [r for r in rows if r["path"] == "verified"]
    return {
        "questions": n,
        "ran_rate": sum(r["ran"] for r in rows) / n,
        "match_rate": sum(r["hit"] for r in rows) / n,
        "match_rate_generated": (sum(r["hit"] for r in gen) / len(gen)) if gen else None,
        "match_rate_verified": (sum(r["hit"] for r in ver) / len(ver)) if ver else None,
        "generated_count": len(gen),
        "verified_count": len(ver),
        "error_count": sum(1 for r in rows if r["path"] == "error"),
    }


def print_report(summary: dict, rows: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"{'questions':<28}{summary['questions']}")
    print(f"{'ran (no error)':<28}{summary['ran_rate']:.0%}")
    print(f"{'match rate (overall)':<28}{summary['match_rate']:.0%}")
    g = summary["match_rate_generated"]
    if g is not None:
        print(f"{'match rate (GENERATED only)':<28}"
              f"{g:.0%}  ← the production number ({summary['generated_count']} q)")
    v = summary["match_rate_verified"]
    if v is not None:
        print(f"{'match rate (verified only)':<28}{v:.0%}  ({summary['verified_count']} q)")
    print("=" * 60)

    print("\nmisses & generated SQL (· ran but wrong, ✗ never ran):")
    for r in rows:
        if not r["hit"]:
            print(f"  {r['id']} [{r['path']}] expect={r['expect']!r}")
            if r.get("sql"):
                print(f"      {r['sql'][:150]}")
            if r.get("error"):
                print(f"      ERROR: {r['error']}")


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
