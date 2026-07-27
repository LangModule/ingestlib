# Limits & scope

What ingestlib deliberately doesn't do, plus the honest edges of what it
does. Trust in the citations depends on being straight about this.

## Scope

- **English documents.** The OCR model reads 100+ languages, but the
  pipeline's prompts, stemmers, and full-text configurations are English.
- **PDF, DOCX, PPTX, and images (PNG/JPEG/WebP) in.** No XLSX — 
  spreadsheets deserve a real tables-first design, not a bad PDF
  conversion; it's on the roadmap.
- **Ingestion is the product; retrieval is a reference.** `retrieve()` is
  a solid hybrid-search + rerank implementation, deliberately not a
  query-pipeline framework (no multi-query, no agents). Everything your
  own reader needs rides on the stored chunk payloads.

## Honest edges

**Unlabeled chart values are estimates.** Printed numbers and callouts are
captured exactly; bar heights without labels are read as estimates and
marked `~` in the data table. No parser can read numbers that aren't
printed.

**Handwriting is out of scope.** The OCR model is unreliable on it, and
page-level frontier-VLM transcription would break the cost model and the
region-level provenance guarantee. Expect near-empty output on
handwritten pages.

**A one-byte change is a new document — but the old version is replaced,
not orphaned.** Content addressing gives v2 a new `doc_id`, and re-ingesting
the file at the same path deletes v1's chunks and artifacts automatically
(logical identity = namespace + path). Old versions never accumulate. What
ingestlib deliberately does *not* keep is version **history** — once
replaced, v1 is gone, not archived. See [Manage a corpus](../how-to/manage-corpus.md).

**Extract's `grounded` flag is text verification, not truth.** It means
the value was found in the cited source text (with numeric normalization,
so `383285.0` matches a printed `383,285`) — it cannot tell you the
document itself is correct, and a derived or reworded value can be right
yet `grounded=False`. Treat ungrounded fields as "verify before use," not
"wrong."

**Local models trail cloud models on dense charts.** The Ollama reference
stack handles the pipeline's judgment tasks well; complex multi-series
charts are where Nova/GPT-5 still win. Test on your own corpus.

**Very large documents are memory-hungry.** Parse holds page renders in
memory for the duration of a document; multi-hundred-page PDFs work but
haven't been optimized. Streaming/checkpointed parsing is roadmap work.

## Deduplication semantics

By design, dedup keys on **file content only** — not on rules or settings.
Ingesting the same bytes with different categories still returns
`skipped`; pass `skip_existing=False` to re-run. A partially-failed ingest
is always retried, because only the manifest — the pipeline's final
write — marks completion.

## Roadmap

Document lifecycle shipped in v1.1 — replace-aware ingestion, folder
[`sync()`](../how-to/manage-corpus.md), and [`backfill()`](../how-to/manage-corpus.md#rebuild-the-vector-store-backfill).
Near-term next: 100-page hardening (streaming/checkpointed parse) and XLSX
input. Watch [GitHub](https://github.com/LangModule/ingestlib) for progress.
