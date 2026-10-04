"""text2SQL samples — questions with expected values over a declared SQL source.

Reuses the eval-sql dataset shape: a YAML with `source:` (a name from sources.yaml)
and `questions:` of {question, expect[, id]}. For a public benchmark (Spider/BIRD),
declare each database as a sqlite source in sources.yaml, then write one dataset
file per database referencing that source name:

    source: spider_concert_singer
    questions:
      - id: q1
        question: how many singers are there?
        expect: "3"
"""
from pathlib import Path

import yaml


def load_sql_dataset(path: str | Path) -> tuple[str, list[dict]]:
    """Return (source_name, questions) from a text2SQL eval YAML."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(
            f"sql dataset not found: {p}\n"
            f"declare the database in sources.yaml and reference its name — "
            f"see evals/datasets/sql.py for the format"
        )
    data = yaml.safe_load(p.read_text()) or {}
    source = data.get("source")
    questions = data.get("questions") or []
    if not source:
        raise ValueError(f"{p} has no `source:` — name a source declared in sources.yaml")
    if not questions:
        raise ValueError(f"{p} has no `questions:` — nothing to evaluate")
    return str(source), questions
