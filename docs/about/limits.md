# Limits & scope

What ingestlib deliberately does not do, and the edges to know about.

## Scope decisions

- **English-only.** Prompts, stemmers, and lexical analyzers are tuned
  for English.
- **PDF / DOCX / PPTX in.** No direct image input (PNG/JPG as files) —
  but images *inside* documents are fully handled (extracted,
  described, embedded as text).
- **No handwriting.** RAG scope is printed documents.
- **Single-tenant by design.** No auth, no job queue, no multi-user
  database. Namespaces partition corpora within one deployment.

## Behavioral edges

- **Unlabeled chart values are estimates.** When a bar has no printed
  number, the enricher estimates from geometry and marks it `~` — no
  parser can read unprinted numbers.
- **Content-addressed IDs**: a one-byte-different file is a new
  document. Version linking ("this replaces that") is app-layer.
- **Page caps**: classify reads at most 100 pages (front matter
  identifies type); split caps at 500; parse has no cap. A try-style
  in-memory run should stay small — page renders are held in memory.
- **Re-parses shift chunk boundaries.** Parse involves an LLM, so two
  parses of the same file can chunk slightly differently; `doc_id`
  (byte hash) is the stable identity, not chunk numbering.
- **Switching `embedding_provider` invalidates existing vectors** —
  see [AI providers](../guides/providers.md) for the safe recipe.

## Operational notes

- The OCR server is the throughput bottleneck (GPU-serialized);
  everything else pipelines behind it.
- Jina's free tier is rate-limited (100 RPM); the client retries with
  backoff and honors `Retry-After`, and reranking degrades to vector
  order rather than failing retrieval.
- Amazon Rerank default quotas are low (2 RPM) — request an increase
  for real workloads.

## Roadmap

Extraction (structured field pulling), per-run rules on `ingest()`,
larger-scale runs (100+ pages end-to-end), and an eval dashboard. See
the [GitHub issues](https://github.com/LangModule/ingestlib/issues).
