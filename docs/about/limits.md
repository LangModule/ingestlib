# Limits & scope

What ingestlib deliberately doesn't do, plus the honest edges of what it
does. Trust in the citations depends on being straight about this.

## Scope

- **English documents.** The OCR model reads 100+ languages, but the
  pipeline's prompts, stemmers, and full-text configurations are English.
- **PDF, DOCX, PPTX, and images (PNG/JPEG/WebP) in.** No XLSX — 
  spreadsheets deserve a real tables-first design, not a bad PDF
  conversion; it's on the roadmap.
- **Ingestion is the product; retrieval is a reference — now over documents
  *and* databases.** `retrieve()` is a solid hybrid-search + rerank
  implementation over the corpus, and with `sources=` it also generates
  read-only SQL over your databases
  ([structured retrieval](../how-to/structured-retrieval.md)). It is
  deliberately not a query-side *framework* — no multi-query planning, no
  agentic reasoning loops of its own. Bring your own reader, or point an agent
  at the [MCP server](../how-to/mcp-server.md); everything either needs rides
  on the stored chunk payloads.

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

**Generated SQL is best-effort; the read-only role is the guarantee.**
Structured retrieval writes read-only SQL from your schema and `tables` hints —
strong on a clean, well-described schema, weaker on a sprawling one with
cryptic columns and heavy joins (the documented reality of text-to-SQL). A
query that *errors* is retried once; one that runs but returns the *wrong*
number looks identical to a right one and can't be caught automatically. Two
things make it safe to point at a real database anyway: a **read-only role**
(the worst case is a wrong read, never a write) and **verified queries** for
answers that must be exact. Measure accuracy on your own schema before trusting
generated numbers — treat them as an analyst draft.

**Local models trail cloud models on dense charts.** The Ollama reference
stack handles the pipeline's judgment tasks well; complex multi-series
charts are where Nova/GPT-5 still win. Test on your own corpus.

**Very large documents are memory-hungry.** Parse holds page renders in
memory for the duration of a document; multi-hundred-page PDFs work but
aren't optimized for memory. Keep single documents to a sane page count,
or split very large files upstream.

## Deduplication semantics

By design, dedup keys on **file content only** — not on rules or settings.
Ingesting the same bytes with different categories still returns
`skipped`; pass `skip_existing=False` to re-run. A partially-failed ingest
is always retried, because only the manifest — the pipeline's final
write — marks completion.

## Roadmap

Recently shipped: [structured retrieval](../how-to/structured-retrieval.md) —
query your SQL databases alongside documents through one `retrieve()` call,
behind a read-only permission boundary (v1.3); an
[MCP server](../how-to/mcp-server.md) to serve the corpus to agents (v1.2);
document lifecycle — replace-aware ingestion, folder
[`sync()`](../how-to/manage-corpus.md), and [`backfill()`](../how-to/manage-corpus.md#rebuild-the-vector-store-backfill) (v1.1).
Near-term next: XLSX input (tables-first, not a PDF conversion). Watch
[GitHub](https://github.com/LangModule/ingestlib) for progress.
