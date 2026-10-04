"""Shared plumbing for the eval suite — load, run, score, snapshot.

Every capability runner follows the same shape: load a dataset of (input, gold)
pairs, run ingestlib over the inputs, score each output against its gold with
that capability's metric, and write a timestamped JSON snapshot to results/.
These helpers hold the parts that don't change between capabilities.

Measures, never asserts — a snapshot records where things stand.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ingestlib.config import get_config

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
# Downloaded public datasets live here (gitignored) — see README.md for the
# per-dataset fetch; only our own gold sets are versioned.
DATA_DIR = EVALS_DIR / "data"


def provider_stamp() -> dict[str, str]:
    """The provider/model fingerprint a score belongs to — a number means
    nothing without the config that produced it."""
    cfg = get_config()
    return {
        "llm_provider": cfg.llm_provider,
        "embedding_provider": cfg.embedding_provider,
    }


def snapshot(name: str, payload: dict[str, Any]) -> Path:
    """Write a timestamped JSON result to results/ and return its path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{name}-{stamp}.json"
    out.write_text(json.dumps({
        **provider_stamp(),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }, indent=2))
    return out


def print_metrics(title: str, metrics: dict[str, float], n: int) -> None:
    """One aggregate metrics block — capability, sample count, each score."""
    print(f"\n{title}  (n={n})")
    for key, value in metrics.items():
        print(f"  {key:<24}{value:>8.3f}")


def mean_metrics(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
    """Mean of each metric across scored rows (every row carries a 'metrics' dict)."""
    n = len(rows) or 1
    return {k: sum(r["metrics"].get(k, 0.0) for r in rows) / n for k in keys}
