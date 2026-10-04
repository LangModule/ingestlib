"""Combined scorecard — the latest snapshot per capability, side by side.

Reads evals/results/*.json and prints one table: each capability's headline
metric, sample count, the provider that produced it, and when it ran. Shows
whatever has been measured so far.

Usage: uv run python -m evals.report
"""
import json

from evals.harness import RESULTS_DIR

# result-file prefix -> (label, headline metric key; None = SQL's summary shape)
_CAPABILITIES = {
    "chart": ("chart→table", "f1"),
    "parse": ("parse", "edit_similarity"),
    "tables": ("tables", "cell_f1"),
    "formulas": ("formulas", "edit_similarity"),
    "sql": ("text2SQL", None),
}


def _latest(prefix: str):
    files = sorted(RESULTS_DIR.glob(f"{prefix}-*.json"))
    return files[-1] if files else None


def main() -> None:
    if not RESULTS_DIR.is_dir() or not any(RESULTS_DIR.glob("*.json")):
        print("no results yet — run an eval first (see evals/README.md)")
        return
    print(f"\n{'capability':<14}{'headline':>10}{'n':>7}  {'provider':<10}ran_at")
    print("-" * 60)
    for prefix, (label, key) in _CAPABILITIES.items():
        latest = _latest(prefix)
        if latest is None:
            print(f"{label:<14}{'—':>10}{'—':>7}")
            continue
        data = json.loads(latest.read_text())
        if prefix == "sql":
            metric = data.get("summary", {}).get("match_rate")
            n = data.get("summary", {}).get("questions", 0)
        else:
            metric = data.get("metrics", {}).get(key)
            n = data.get("samples", 0)
        shown = f"{metric:.3f}" if isinstance(metric, (int, float)) else "—"
        provider = data.get("llm_provider", "?")
        ran_at = str(data.get("ran_at", ""))[:19]
        print(f"{label:<14}{shown:>10}{n:>7}  {provider:<10}{ran_at}")


if __name__ == "__main__":
    main()
