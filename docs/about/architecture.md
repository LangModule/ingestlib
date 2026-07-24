# Architecture

```
services      ingest · retrieve            the composed flows
operations    parse · classify · split     each usable standalone
storage       artifact store (s3 | local) + 8 vector-store connectors
foundations   LLM providers (bedrock | openai) + OCR engine (PaddleOCR-VL)
```

Four layers; every layer only calls downward. The rules that shape the
codebase:

## One engine, one judgment LLM

Parsing uses exactly two models with a sharp division of labor: a small
local vision-language model (PaddleOCR-VL-1.6) for layout-aware reading
— which it does at state-of-the-art accuracy — and a frontier LLM only
for what the small model verifiably gets wrong: chart values, figure
descriptions, and a per-region review pass. All *judgment* in the
pipeline (classification, section vocabulary, chunk boundaries) goes
through the same single LLM, behind the provider switch.

## Provenance is load-bearing

Every parsed block carries a `region_id` and bounding box; every chunk
carries `region_ids` per page; every retrieval hit resolves back to
exact regions on exact pages. The review pass returns *per-region*
corrections rather than rewriting pages precisely so this chain never
breaks. Citations are a data structure, not a string format.

## The artifact store is the source of truth

Every stage's output is persisted under `documents/{doc_id}/` (the
content hash). Vector stores are derived indexes — rebuildable from
stored splits at any time, which is what makes switching connectors or
embedding providers a backfill rather than a re-parse.

## Explicit writes, lazy configuration

Nothing writes to storage behind the caller's back — persistence happens
in `ingest()` or explicit `artifacts.save_*` calls. Nothing reads
configuration at import time — the first library call discovers
config.yaml, and `reset_config()` reloads it in-process (this is what
lets the studio apply settings edits with zero restarts).

## Guarantees live in code, not prompts

Where correctness matters — tables never split, captions bound to their
figures, chunk-size ceilings, valid section partitions, closed-set
labels — an LLM proposes and Python enforces. Prompts are suggestions;
the post-processing is the contract.
