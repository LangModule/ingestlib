# Architecture

Four layers, strict downward dependencies — nothing lower knows what sits
above it.

```text
src/ingestlib/
├── services/       ingest · retrieve · lifecycle (remove · sync · backfill) — the product
├── operations/     parse · classify · split · extract — the tools (each standalone)
├── storage/        artifacts (S3 | local) · VectorStore contract · 8 connectors
├── foundations/    llm (Bedrock · OpenAI · Ollama · Jina rerank) · ocr (PaddleOCR-VL)
├── cli/            the `ingestlib` command — init · doctor · ingest · sync · list · remove · backfill · search
├── utils/          logger · files · sync · aws
└── config.py       config.yaml + .env + rules.yaml → typed, frozen configs
```

## The load-bearing decisions

**Provider dispatch is a per-call config read.** Operations import
`chat`/`embed_text` from one surface; which backend answers is decided at
call time. No client is built until a call happens, and backends that
aren't selected are never imported — a sqlite + ollama pipeline never
touches boto3.

**The VectorStore contract absorbs backend quirks.** ID schemes, metadata
encoding, fusion mechanics, deletion semantics — each connector handles
its backend's reality internally (documented at the top of each module) so
pipelines are written once. Shared guarantees: idempotent upserts, orphan
pruning on re-ingest, no infrastructure creation on the read path,
namespace isolation everywhere.

**Artifacts are the source of truth; vectors are an index.** Every stage's
output persists before the next stage runs, so nothing about a corpus is
ever locked inside a vector database — parses, chunks, and page renders
all reload from the artifact store. `backfill()` rebuilds a vector store
straight from these artifacts, no re-parse.

**Provenance is structural, not annotated.** Chunks record the parse
region ids they cover, chunk boundaries can't cut through a region, and
the full payload rides on the vector. The citation chain
(hit → regions → bboxes → page render) needs no extra database.

**Errors carry their fix.** Every backend boundary translates its classic
failures — wrong AWS profile, missing model access, dead Ollama, exhausted
Jina quota — into one-sentence remedies, re-raised with the original
chained. `ingestlib doctor` is those same translations, run proactively.

**Config is discovered at call time, never at import.** Importing
ingestlib does nothing; the first real call finds `config.yaml` (explicit
env var, else CWD and parents). Frozen dataclasses make a loaded config
immutable; changing files mid-process takes an explicit reset.

## Testing philosophy

Real APIs, never mocks. Pure logic runs on every test invocation;
server-hitting suites are opt-in via `RUN_*_E2E` gates. Failures are
tested by *provoking real ones* — bogus keys get real 401s, dead ports get
real connection refusals — so the error translations are verified against
reality, not against a mock's guess. The sqlite connector's full suite
runs ungated: there is no server, so in-process *is* the real thing.
