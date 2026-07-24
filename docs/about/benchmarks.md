# Benchmarks

Measured, not asserted: the repository ships a retrieval eval harness
(`evals/` + `make eval`) that runs the full pipeline and scores
retrieval against hand-built ground truth.

## Methodology

- 22 questions over 12 real fixture documents (earnings decks,
  clinical studies, government forms, timetables — 29 pages read
  personally to build ground truth).
- Ground truth is `(document, pages, keywords)` — never chunk IDs, so
  the eval survives re-parses that change chunk boundaries.
- Metrics: hit@1 / hit@3 / hit@5 and MRR, per configuration
  (dense-only, hybrid, each ± rerank).

## Results

| Configuration | hit@3 | hit@5 | MRR |
|---|---|---|---|
| **Hybrid + rerank** (the default) | **1.00** | **1.00** | ~0.98 |
| Dense + rerank | 1.00 | 1.00 | ~0.95–0.98 |
| Dense or hybrid alone | ~0.86 hit@1-range | | |

Identical scores across every connector tested — SQLite matches the
clouds, which is why it is a first-class recommendation and not a toy.

## What the numbers taught us

- **Reranking is the biggest lever**: +5 to +14 points hit@1 over raw
  vector order. It rescues table-content questions that dense search
  misses outright.
- **Hybrid earns its keep with the reranker**: lexical candidates fix
  exact-token questions (names, part numbers, printed percentages) that
  dense embedding smooths away; the reranker then arbitrates between
  the two signals.
- **hit@1 is noisy run-to-run** (~0.86–1.00): parse is LLM-driven, so a
  re-parse shifts chunk boundaries. The stable headline metric is
  hit@3.
- **Evals catch parse bugs**: a missing chart-annotation callout
  ("+360%") made one question unanswerable; the fix was verified by the
  eval going from unfindable to rank 1.

## Honest caveats

- The corpus is small (~35 chunks) — ceiling effects are real, and the
  numbers should be re-measured as corpora grow.
- Parse speed: ~12 s/page all-in on an M-series Mac (OCR-bound). LLM
  cost: ~$0.002/page.
