"""Parse eval — assembled page markdown vs gold, over a doc manifest.

Runs ingestlib's parse over each document and scores its whole-document markdown
against the gold with normalized edit similarity + exact match. Measures, never
asserts. Needs the OCR server (GPU).

Usage:
    uv run python -m evals.run_parse_eval --dataset evals/data/omnidocbench/manifest.jsonl
    uv run python -m evals.run_parse_eval --dataset ... --limit 50 --concurrency 4
"""
import argparse
import asyncio

from evals.adapters.parse import predict_markdown
from evals.datasets.docs import load_manifest
from evals.harness import mean_metrics, print_metrics, snapshot
from evals.scorers.text import score_text

_AGG_KEYS = ("edit_similarity", "exact")


async def _run_one(sample, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            pred = await predict_markdown(sample.doc_path)
        except Exception as exc:
            return {"id": sample.id, "error": str(exc), "metrics": dict.fromkeys(_AGG_KEYS, 0.0)}
    return {"id": sample.id, "metrics": score_text(pred, sample.gold_markdown)}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="parse JSONL manifest")
    parser.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    samples = load_manifest(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
    print(f"parse: {len(samples)} docs from {args.dataset} (needs the OCR server)")

    sem = asyncio.Semaphore(args.concurrency)
    rows = list(await asyncio.gather(*(_run_one(s, sem) for s in samples)))

    agg = mean_metrics(rows, _AGG_KEYS)
    errors = sum(1 for r in rows if "error" in r)
    print_metrics("parse (page markdown)", agg, len(rows))
    if errors:
        print(f"  ({errors} sample(s) errored)")

    out = snapshot("parse", {
        "dataset": args.dataset,
        "samples": len(rows),
        "metrics": {**agg, "errors": float(errors)},
        "per_sample": rows,
    })
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
