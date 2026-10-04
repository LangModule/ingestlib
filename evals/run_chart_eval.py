"""Chart→table eval — the enricher's chart reading, scored on public charts.

Runs ingestlib's chart-enrichment path over a chart manifest (ChartX / PlotQA
converted to the harness format), scores each predicted table against its gold
with numeric F1 + RNSS, and snapshots the aggregate. Measures, never asserts.
GPU-free — needs only the configured LLM.

Usage:
    uv run python -m evals.run_chart_eval --dataset evals/data/chartx/manifest.jsonl
    uv run python -m evals.run_chart_eval --dataset ... --limit 100 --concurrency 8
"""
import argparse
import asyncio
from pathlib import Path

from evals.adapters.chart import predict_chart_table
from evals.datasets.charts import load_manifest
from evals.harness import mean_metrics, print_metrics, snapshot
from evals.scorers.rms import score_chart

_AGG_KEYS = ("precision", "recall", "f1", "rnss")


def _image_format(path: Path) -> str:
    return "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"


async def _run_one(sample, sem: asyncio.Semaphore, rel_tol: float) -> dict:
    async with sem:
        try:
            pred = await predict_chart_table(
                sample.image_path.read_bytes(), _image_format(sample.image_path)
            )
        except Exception as exc:
            return {"id": sample.id, "type": sample.chart_type, "error": str(exc),
                    "metrics": dict.fromkeys(_AGG_KEYS, 0.0)}
    return {
        "id": sample.id,
        "type": sample.chart_type,
        "prediction": pred,
        "metrics": score_chart(pred, sample.gold_table, rel_tol),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="chart→table JSONL manifest")
    parser.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rel-tol", type=float, default=0.05,
                        help="relative tolerance for a numeric match (default 5%)")
    args = parser.parse_args()

    samples = load_manifest(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    print(f"chart→table: {len(samples)} charts from {args.dataset}")

    sem = asyncio.Semaphore(args.concurrency)
    rows = list(await asyncio.gather(*(_run_one(s, sem, args.rel_tol) for s in samples)))

    agg = mean_metrics(rows, _AGG_KEYS)
    errors = sum(1 for r in rows if "error" in r)
    print_metrics("chart→table", agg, len(rows))
    if errors:
        print(f"  ({errors} sample(s) errored)")

    out = snapshot("chart", {
        "dataset": args.dataset,
        "rel_tol": args.rel_tol,
        "samples": len(rows),
        "metrics": {**agg, "errors": float(errors)},
        "per_sample": rows,
    })
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
