"""Table eval — recovered table HTML vs gold, over a table manifest.

Runs ingestlib's parse over each table image and scores the recovered table
against the gold with cell-content F1 + a shape match. Measures, never asserts.
Needs the OCR server (GPU).

Usage:
    uv run python -m evals.run_table_eval --dataset evals/data/fintabnet/manifest.jsonl
"""
import argparse
import asyncio

from evals.adapters.table import predict_table_html
from evals.datasets.tables import load_manifest
from evals.harness import mean_metrics, print_metrics, snapshot
from evals.scorers.teds import score_table

_AGG_KEYS = ("cell_precision", "cell_recall", "cell_f1", "shape_match")


async def _run_one(sample, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            pred = await predict_table_html(sample.image_path)
        except Exception as exc:
            return {"id": sample.id, "error": str(exc), "metrics": dict.fromkeys(_AGG_KEYS, 0.0)}
    return {"id": sample.id, "metrics": score_table(pred, sample.gold_table)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="table JSONL manifest")
    parser.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    samples = load_manifest(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    print(f"tables: {len(samples)} tables from {args.dataset} (needs the OCR server)")

    sem = asyncio.Semaphore(args.concurrency)
    rows = list(await asyncio.gather(*(_run_one(s, sem) for s in samples)))

    agg = mean_metrics(rows, _AGG_KEYS)
    errors = sum(1 for r in rows if "error" in r)
    print_metrics("tables (cell F1)", agg, len(rows))
    if errors:
        print(f"  ({errors} sample(s) errored)")

    out = snapshot("tables", {
        "dataset": args.dataset,
        "samples": len(rows),
        "metrics": {**agg, "errors": float(errors)},
        "per_sample": rows,
    })
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
