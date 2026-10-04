"""text2SQL eval — question/expect over a declared SQL source (Spider / BIRD).

Runs ingestlib's answer() over the questions and value-matches the executed rows,
reusing the eval-sql CLI's scoring. Measures, never asserts. GPU-free (needs the DB).

Declare each benchmark database as a sqlite source in sources.yaml and point at a
dataset file that names it — see evals/datasets/sql.py.

Usage:
    uv run python -m evals.run_sql_eval --dataset evals/data/spider/concert_singer.yaml
"""
import argparse
import asyncio
import os

from ingestlib.cli.eval_sql import print_report, run, summarize
from ingestlib.sources.registry import resolve_sources
from ingestlib.sources.sql.engine import reset_engines
from ingestlib.utils.logger import configure

from evals.datasets.sql import load_sql_dataset
from evals.harness import snapshot


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="text2SQL eval YAML (source + questions)")
    args = parser.parse_args()

    if not os.environ.get("INGESTLIB_LOG_LEVEL"):
        configure(level="WARNING")

    source_name, questions = load_sql_dataset(args.dataset)
    [source] = resolve_sources([source_name])
    if source.__class__.__name__ != "SqlSource":
        raise SystemExit(f"source {source_name!r} is not a SQL database")

    print(f"text2SQL: {len(questions)} questions against {source_name}")
    reset_engines()
    try:
        rows = await run(source, questions)
    finally:
        reset_engines()

    summary = summarize(rows)
    print_report(summary, rows)
    out = snapshot("sql", {
        "source": source_name,
        "dataset": args.dataset,
        "summary": summary,
        "per_question": rows,
    })
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
