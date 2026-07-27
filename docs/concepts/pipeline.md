# The pipeline

ingestlib is four **operations** and two **services**. Each operation
works standalone; the services chain the first three and add persistence.
The fourth, `extract()`, is a standalone reader — it pulls structured
data out of a document and is not part of the ingest chain. Around the two
services sit the corpus-management helpers — `remove`, `sync`, `backfill` —
that keep the stores in step with changing files
([Manage a corpus](../how-to/manage-corpus.md)).

```text
                     ┌───────────────── services ─────────────────┐
                     │                                            │
  report.pdf ──▶  ingest():  parse ─▶ classify ─▶ split ─▶ embed ─▶ upsert
                     │                                            │
  "question"  ──▶ retrieve():  embed ─▶ vector search ─▶ rerank ─▶ cited hits
```

## Parse — the only stage that reads pixels

`parse()` renders each page and runs it through PaddleOCR-VL for layout
detection and recognition — text, tables (HTML with merged cells), formulas
(LaTeX), charts, seals. The LLM then enriches what OCR alone can't:

- **Charts become data tables** — printed values captured exactly,
  estimated bar heights marked `~`
- **Figures get descriptions** and are cropped out as PNGs
- **A review pass** corrects region-level mistakes

The result is a `ParseResult`: per-page markdown plus every region's
bounding box. Parse is the expensive stage (OCR server + LLM per page) and
the only one that needs the OCR server.

Scanned documents are first-class: parse always OCRs the page *render* and
never trusts the file's text layer.

## Classify — cheap, no OCR

`classify()` reads the document's native text and embedded images (or a
`ParseResult` if you have one) and returns a document-type label with
confidence and ranked alternatives. Open-ended by default; constrained to
your own categories when you provide [content rules](../how-to/content-rules.md).

Documents up to 20 pages classify in a single call; larger ones map-reduce
over 20-page windows.

## Split — sections, then natural chunks

`split()` works in three passes:

1. **Section vocabulary** — the LLM proposes the roles pages play in *this*
   document (`methods`, `results`, …) — skipped when you supply your own
   vocabulary
2. **Page labeling** — every page assigned to a section
3. **Chunking** — within each section, boundaries follow the content:
   headings, topic shifts, table edges. Tables and figures are never split.

Each chunk records its pages and the parse region ids it covers, and
carries a `[category › section › heading]` breadcrumb in the text that
gets embedded.

## Extract — your schema, grounded

`extract()` fills a Pydantic schema you define from the document — one
instance for the whole document (`mode="one"`), or every instance it can
find (`mode="many"`: all the receipts in an expense bundle). It is the
operation that returns *data* rather than document structure, and its
results are verified, not just generated:

- Every field cites the **page and parse regions** it was read from
- Each value is **grounded** — checked against the cited source text
- Confidence is honest: uncited or ungrounded fields are capped, and
  citations pointing at regions that don't exist are dropped

Feed it a `ParseResult` for region-level citations on scans; feed it a
raw path and it reads the native text layer with page-level citations
and no OCR server. See [Extract structured data](../how-to/extract.md).

## Ingest — the pipeline with persistence

`ingest()` chains parse, classify, and split, embeds every chunk, and upserts
into the configured vector store — persisting each stage's output to the
artifact store as it goes. Content-checksum dedup makes it safe to point
at the same folder twice. A progress callback (`on_stage`) reports each
stage's start/finish.

## Retrieve — the reference reader

`retrieve()` embeds the question, queries the vector store (hybrid stores
also run a lexical search and merge), optionally reranks the candidate
pool with a cross-encoder, and returns `Hit`s with scores, citations, and
a prompt-ready `context` block.

Retrieval here is a solid reference implementation — the ingestion side is
where ingestlib invests. Bring your own query pipeline if you have one;
everything it needs is on the stored chunk payloads.

## Standalone composition

The services are convenience, not lock-in:

```python
from ingestlib.operations import parse, classify, split, extract

result = parse("report.pdf")               # or load_parse(doc_id) from artifacts
label  = classify(result)                  # reuses the parse — no OCR
chunks = split(result, category=label.category)
fields = extract(result, schema=MySchema)  # reuses the parse — cited fields
```

`classify`, `split`, and `extract` also accept raw paths directly — they
read native text without OCR, and raise a clear error telling you to
`parse()` first when a document has no text layer (scans, images).

---

Next: [Provenance & citations](provenance.md) — the chain that makes
answers point at pages.
