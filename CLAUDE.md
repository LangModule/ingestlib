# ingestlib

Self-hosted document intelligence for RAG — parse, classify, and split
PDF/DOCX/PPTX into searchable, cited, retrieval-ready chunks. Published on
PyPI (`pip install ingestlib`). The visual review app lives in the sibling
repo `../ingestlib-studio`.

## Architecture

Four layers, strict downward dependencies:

```
services/       ingest · retrieve            — the product
operations/     parse · classify · split     — the tools (each standalone)
storage/        artifacts (S3) · base (VectorStore) · pinecone · qdrant
foundations/    llm (Bedrock Nova, Jina) · ocr (PaddleOCR-VL)
```

Core principles: operations work standalone AND compose; **LLMs propose,
code guarantees** (invariants live in deterministic code); provenance
everywhere (every chunk traces to region_ids + bboxes on page images);
doc_id = SHA-256 of file bytes; buckets/indexes/collections bootstrap on
first use.

The full design doc is `plan.md` (gitignored — local only). Read it before
making architectural changes.

## Commands

```bash
make test          # fast suite (~180 tests, ~90s) — e2e groups skip
make test-all      # everything (needs VL server + full stack)
make eval          # retrieval quality measurement (evals/, NOT a test)
uv run ruff check src/ tests/ evals/
```

The OCR inference server must run for parse work:

```bash
uv run python -m mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6
```

## Conventions

- **Tests hit real APIs, never mocks.** Pure logic always runs; anything
  touching a server is opt-in via `RUN_*_E2E=1` gates (strict `!= "1"`).
  In-process/embedded modes of vendors are NOT equivalent to their real
  servers — verify against the real thing at least once.
- **Evals measure, tests assert.** Quality numbers live in `evals/` with
  timestamped snapshots; never turn them into hard CI assertions.
- Config loads lazily; discovery is `INGESTLIB_CONFIG` env var → config.yaml
  in CWD/parents. `config.yaml` and `.env` are gitignored; `.example`
  variants are tracked. Never commit personal AWS values.
- Python 3.12+, ruff, line-length 100. Frozen models everywhere
  (dataclasses.replace / model_copy to mutate).
- Comments state constraints the code can't show — no narration.
- The S3 bucket name `ingestlib` is globally claimed by us: empty it if
  needed, never delete the bucket itself.
