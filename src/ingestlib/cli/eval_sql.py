"""`ingestlib eval-sql NAME` — measure text2SQL accuracy on YOUR schema.

Measures, never asserts (like the retrieval eval). The generate path is
analyst-assist, not autonomous-trusted: the only way to know how far to trust a
generated number on a given schema is to measure it there. This runs a YAML
question/expected set through the real answer() path of a declared SQL source and
reports the overall match rate, the GENERATED-only rate (the honest production
number, with no verified query to fall back on), and the verified/generated split.

Build the set on your own schema, re-run it after changing hints, schema_rag, or
the model, and keep "show the SQL before trusting the number" as the rule. Text2SQL
accuracy drifts with the model and the hints — a report informs; a red CI run would
just block you.

Dataset (YAML): a list under `questions:` of {question, expect[, id]} — expect is
one value or a list; a run hits when any expected value appears in the result
(numbers matched exactly so "5" never matches "25"; text case-insensitive
substring). Defaults to <source>_eval.yaml beside sources.yaml; override with
--dataset. See evals/sql_dataset.yaml for the format.
"""
from ingestlib.utils.logger import get_logger

logger = get_logger(__name__)


def _cells(rows: list[tuple]) -> list[str]:
    """Every non-null value in the result, stringified and stripped — the haystack."""
    return [str(v).strip() for row in rows for v in row if v is not None]


def matched(expect, rows: list[tuple]) -> bool:
    """A hit when any expected value is present: a number must match a cell exactly
    (so "5" never matches "25"), text matches a cell case-insensitively as a
    substring, and a list of expects hits if ANY one appears."""
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


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    gen = [r for r in rows if r["path"] == "generated"]
    ver = [r for r in rows if r["path"] == "verified"]
    return {
        "questions": n,
        "ran_rate": sum(r["ran"] for r in rows) / n if n else 0.0,
        "match_rate": sum(r["hit"] for r in rows) / n if n else 0.0,
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
            print(f"  {r.get('id', r['question'][:40])} [{r['path']}] expect={r['expect']!r}")
            if r.get("sql"):
                print(f"      {r['sql'][:150]}")
            if r.get("error"):
                print(f"      ERROR: {r['error']}")


async def run(source, dataset: list[dict]) -> list[dict]:
    """Run every question through answer(), scoring execution-match. Never raises
    on a bad query — a guard reject or DB error is recorded as a non-run miss."""
    rows_out = []
    for q in dataset:
        qid = q.get("id", q["question"][:40])
        entry = {"id": qid, "question": q["question"], "expect": q["expect"]}
        try:
            [result] = await source.answer(q["question"])
            entry["ran"] = True
            entry["path"] = "verified" if result.provenance.get("verified") else "generated"
            entry["sql"] = result.provenance.get("sql", "")
            entry["hit"] = matched(q["expect"], result.raw["rows"])
        except Exception as exc:
            entry.update(ran=False, path="error", sql="", hit=False, error=str(exc)[:200])
        rows_out.append(entry)
        mark = "✓" if entry["hit"] else ("·" if entry["ran"] else "✗")
        print(f"  {mark} [{entry['path']:<9}] {qid}")
    return rows_out


def _load_dataset(source_name: str, dataset: str | None) -> list[dict]:
    from pathlib import Path

    import yaml

    from ingestlib.config import _find_config_path

    if dataset:
        path = Path(dataset).expanduser()
    else:
        path = _find_config_path().parent / f"{source_name}_eval.yaml"
    if not path.is_file():
        raise ValueError(
            f"no eval dataset at {path} — pass --dataset PATH, or create it as a YAML "
            f"list under `questions:` of {{question, expect}} (see evals/sql_dataset.yaml)"
        )
    data = yaml.safe_load(path.read_text()) or {}
    questions = data.get("questions") or []
    if not questions:
        raise ValueError(f"{path} has no `questions:` — nothing to evaluate")
    return questions


def run_eval_sql(source_name: str, *, dataset: str | None = None) -> int:
    import asyncio
    import os

    from ingestlib.sources.registry import resolve_sources
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.utils.logger import configure

    if not os.environ.get("INGESTLIB_LOG_LEVEL"):
        configure(level="WARNING")

    questions = _load_dataset(source_name, dataset)
    [source] = resolve_sources([source_name])
    if source.__class__.__name__ != "SqlSource":
        raise ValueError(f"source {source_name!r} is not a SQL database")

    print(f"running {len(questions)} text2SQL questions against {source_name} ...")
    reset_engines()
    try:
        rows = asyncio.run(run(source, questions))
    finally:
        reset_engines()
    print_report(summarize(rows), rows)
    return 0
