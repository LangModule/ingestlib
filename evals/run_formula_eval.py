"""Formula eval — recovered LaTeX vs gold, over a formula manifest.

Runs ingestlib's parse over each formula image and scores the recovered LaTeX
against the gold with normalized edit similarity + exact match. Measures, never
asserts. Needs the OCR server (GPU).

Usage:
    uv run python -m evals.run_formula_eval --dataset evals/data/unimer/manifest.jsonl
"""
import argparse
import asyncio

from evals.adapters.formula import predict_latex
from evals.datasets.formulas import load_manifest
from evals.harness import mean_metrics, print_metrics, snapshot
from evals.scorers.text import score_text

_AGG_KEYS = ("edit_similarity", "exact")


async def _run_one(sample, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            pred = await predict_latex(sample.image_path)
        except Exception as exc:
            return {"id": sample.id, "error": str(exc), "metrics": dict.fromkeys(_AGG_KEYS, 0.0)}
    return {"id": sample.id, "metrics": score_text(pred, sample.gold_latex)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="formula JSONL manifest")
    parser.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    samples = load_manifest(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    print(f"formulas: {len(samples)} formulas from {args.dataset} (needs the OCR server)")

    sem = asyncio.Semaphore(args.concurrency)
    rows = list(await asyncio.gather(*(_run_one(s, sem) for s in samples)))

    agg = mean_metrics(rows, _AGG_KEYS)
    errors = sum(1 for r in rows if "error" in r)
    print_metrics("formulas (LaTeX)", agg, len(rows))
    if errors:
        print(f"  ({errors} sample(s) errored)")

    out = snapshot("formulas", {
        "dataset": args.dataset,
        "samples": len(rows),
        "metrics": {**agg, "errors": float(errors)},
        "per_sample": rows,
    })
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
