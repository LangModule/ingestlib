"""Retrieval quality eval — measures, never asserts.

Runs every dataset question through the real retrieve() flow under the 2x2
grid {dense, hybrid} x {no-rerank, rerank}, scores hit@k and MRR against
(doc, pages, keywords) ground truth, prints a comparison table, and saves a
timestamped snapshot to evals/results/.

This is a measurement harness, not a test: quality numbers drift with
models, chunking, and data — a report tells you, a red CI run blocks you.

Usage:
    uv run python evals/run_eval.py                # ensure corpus ingested, run all configs
    uv run python evals/run_eval.py --skip-ingest  # corpus already ingested
    uv run python evals/run_eval.py --store qdrant   # or sqlite | pgvector | mongodb | milvus
    uv run python evals/run_eval.py --store sqlite --backfill   # fresh/wiped store:
                                   # re-embed S3 split artifacts into it (no VL server)
    uv run python evals/run_eval.py --top-k 5
"""
import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ingestlib.foundations.llm import aembed_text
from ingestlib.services.ingest.ingestor import aingest
from ingestlib.services.retrieve.retriever import aretrieve
from ingestlib.storage import (
    MilvusStore,
    MongodbStore,
    PgvectorStore,
    PineconeStore,
    QdrantStore,
    SqliteStore,
    VectorStore,
    artifacts,
)
from ingestlib.utils.files import sha256_of_file

EVALS_DIR = Path(__file__).resolve().parent
PDF_DIR = EVALS_DIR.parent / "tests" / "data" / "pdf"
RESULTS_DIR = EVALS_DIR / "results"

# (label, hybrid, rerank)
CONFIGS = [
    ("dense", False, False),
    ("dense+rerank", False, True),
    ("hybrid", True, False),
    ("hybrid+rerank", True, True),
]

STORES: dict[str, type[VectorStore]] = {
    "pinecone": PineconeStore,
    "qdrant": QdrantStore,
    "sqlite": SqliteStore,
    "pgvector": PgvectorStore,
    "mongodb": MongodbStore,
    "milvus": MilvusStore,
}


def load_dataset() -> list[dict]:
    """Ground-truth entries from dataset.yaml — ids must be unique (they key
    the per-question rank maps in every snapshot)."""
    entries = yaml.safe_load((EVALS_DIR / "dataset.yaml").read_text())
    ids = [e["id"] for e in entries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise SystemExit(f"dataset.yaml has duplicate ids: {duplicates}")
    return entries


async def ensure_ingested(pdfs: list[Path], store: VectorStore) -> None:
    """Ingest any fixture not yet in the artifact store (checksum-skipped otherwise).

    Only checks S3 artifacts — it cannot see whether the *vector store* has
    the corpus. Pointing at a store the corpus was never upserted into
    (fresh sqlite file, wiped index) needs --backfill.
    """
    for pdf in pdfs:
        doc_id = sha256_of_file(pdf)
        if artifacts.document_exists(doc_id):
            continue
        print(f"  ingesting {pdf.name} (needs the VL inference server) ...")
        result = await aingest(pdf, store=store)
        print(f"    -> {result.status}: {result.chunks} chunks in {result.total_seconds:.0f}s")


async def backfill_store(store: VectorStore) -> None:
    """Re-embed every document's S3 split artifact into the store.

    Parse/classify/split are reused from S3 — only embedding (Bedrock) and
    upsert run, so no VL server is needed. Upserts are idempotent, so
    re-backfilling an already-populated store is safe.
    """
    semaphore = asyncio.Semaphore(8)

    async def embed(text: str) -> list[float]:
        async with semaphore:
            return await aembed_text(text)

    for meta in artifacts.list_documents():
        chunks = artifacts.load_split(meta.doc_id).chunks
        embeddings = list(await asyncio.gather(*[embed(c.embedding_text) for c in chunks]))
        store.upsert_chunks(meta.doc_id, chunks, embeddings, category=meta.category)
        print(f"  backfilled {meta.filename}: {len(chunks)} chunks ({meta.category})")


def is_hit(hit, expected_doc_id: str, pages: list[int], keywords: list[str]) -> bool:
    """Ground-truth match: right document, page overlap (when given), and at
    least one keyword in the chunk's markdown+text (case-insensitive)."""
    chunk = hit.chunk
    if chunk.document_id != expected_doc_id:
        return False
    if pages and not set(chunk.pages) & set(pages):
        return False
    if keywords:
        haystack = f"{chunk.markdown}\n{chunk.text}".lower()
        if not any(k.lower() in haystack for k in keywords):
            return False
    return True


async def eval_config(
    label: str,
    store: VectorStore,
    rerank: bool,
    dataset: list[dict],
    doc_ids: dict[str, str],
    top_k: int,
) -> dict:
    """Run every question through retrieve(); return metrics + per-question ranks."""
    per_question: dict[str, int | None] = {}
    for entry in dataset:
        result = await aretrieve(entry["question"], top_k=top_k, rerank=rerank, store=store)
        expected = doc_ids[entry["doc"]]
        rank = None  # 1-indexed rank of the first correct chunk
        for i, hit in enumerate(result.hits, start=1):
            if is_hit(hit, expected, entry.get("pages", []), entry.get("keywords", [])):
                rank = i
                break
        per_question[entry["id"]] = rank

    n = len(dataset)
    ranks = [r for r in per_question.values() if r is not None]
    # dedup keeps the table sane when top_k collides with 1 or 3
    ks = sorted({k for k in (1, 3, top_k) if k <= top_k})
    metrics = {f"hit@{k}": sum(1 for r in ranks if r <= k) / n for k in ks}
    metrics["mrr"] = sum(1.0 / r for r in ranks) / n
    return {"config": label, "metrics": metrics, "ranks": per_question}


def print_table(results: list[dict]) -> None:
    """Config comparison table, then per-question ranks for every question
    at least one config missed or ranked below 1 — the debugging view."""
    cols = list(results[0]["metrics"])
    print(f"\n{'config':<16}" + "".join(f"{c:>9}" for c in cols))
    for r in results:
        print(f"{r['config']:<16}" + "".join(f"{r['metrics'][c]:>9.2f}" for c in cols))

    print("\nmisses (question: rank per config, - = not found):")
    all_ids = results[0]["ranks"].keys()
    for qid in all_ids:
        ranks = [r["ranks"][qid] for r in results]
        if any(rank is None or rank > 1 for rank in ranks):
            cells = " ".join(f"{r['config']}={rank or '-'}" for r, rank in zip(results, ranks))
            print(f"  {qid}: {cells}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", choices=sorted(STORES), default="pinecone")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-ingest", action="store_true",
                        help="skip the corpus-completeness check (no VL server needed)")
    parser.add_argument("--backfill", action="store_true",
                        help="re-embed S3 split artifacts into the selected store — for a "
                             "store the corpus was never upserted into (no VL server needed)")
    args = parser.parse_args()

    dataset = load_dataset()
    pdfs = [PDF_DIR / entry["doc"] for entry in dataset]
    missing = [p.name for p in pdfs if not p.exists()]
    if missing:
        raise SystemExit(f"dataset references missing fixtures: {missing}")
    doc_ids = {p.name: sha256_of_file(p) for p in set(pdfs)}

    store_cls = STORES[args.store]
    if not args.skip_ingest:
        print(f"ensuring corpus is ingested into {args.store} ...")
        await ensure_ingested(sorted(set(pdfs)), store_cls())
    if args.backfill:
        print(f"backfilling {args.store} from S3 split artifacts ...")
        await backfill_store(store_cls())

    results = []
    t0 = time.perf_counter()
    for label, hybrid, rerank in CONFIGS:
        print(f"running {label} ({len(dataset)} questions) ...")
        results.append(await eval_config(
            label, store_cls(hybrid=hybrid), rerank, dataset, doc_ids, args.top_k,
        ))
    duration = time.perf_counter() - t0

    print_table(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"eval-{args.store}-{stamp}.json"
    out.write_text(json.dumps({
        "store": args.store,
        "top_k": args.top_k,
        "questions": len(dataset),
        "duration_seconds": round(duration, 1),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }, indent=2))
    print(f"\nsaved {out.relative_to(EVALS_DIR.parent)} ({duration:.0f}s total)")


if __name__ == "__main__":
    asyncio.run(main())
