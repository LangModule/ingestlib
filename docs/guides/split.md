# Split

Group a document's pages into role-based **sections**, then cut each
section into natural, retrieval-ready **chunks**.

```python
from ingestlib.operations import split

result = split(parse_result, category="research_paper")
for section in result.sections:
    print(section.name, section.pages)        # study_design [2, 3, 4]
for chunk in result.chunks:
    print(chunk.heading, chunk.token_estimate)
```

Like classify, split is dual-input: a `ParseResult` (block-level
provenance) or a raw file path (native text, page-level provenance).
Async: `asplit`. Page cap: 500.

## Sections vs chunks — two different things

- A **section** is a run of consecutive pages sharing a role:
  "methods", "results", "financial_statements". Sections answer *what
  is where* in the document.
- A **chunk** is what gets embedded: a topically coherent piece of a
  section, sized for retrieval. Chunks answer *what can be found*.

## The three passes

1. **Vocabulary** — one LLM call proposes 2–15 snake_case section
   categories for the document. Role-based names are enforced
   ("methods", "results"); layout names ("tables", "figures") are
   forbidden by prompt.
2. **Page labels** — every page picks from that fixed vocabulary, in
   parallel; invalid answers inherit the left neighbor. Consecutive
   same-label pages become `Section`s in pure Python.
3. **Chunk boundaries** — per section, the LLM groups blocks into
   topical chunks with headings; code then enforces the guarantees
   below.

### The chunk guarantees (enforced in code, not prompts)

- A table is never split.
- A caption and its figure stay together.
- A heading never ends a chunk — it binds to the content below it.
- Micro-chunks are merged before size enforcement.
- `max_chunk_tokens` (default 768) is a hard ceiling: an oversized
  group gets one more LLM call proposing budget-aware sub-boundaries
  (the cut lands where the topic pauses), and a greedy walk over block
  boundaries is the final guarantee. A single giant block (one huge
  table) stays whole rather than being cut mid-table.

## Your own categories

Provide the vocabulary yourself and pass 1 is skipped entirely — one
fewer LLM call and deterministic section names:

```python
result = split(parse_result, vocabulary={
    "financial_statements": "Balance sheets, income statements, cash flow",
    "notes": "Footnotes and disclosures",
}, unmatched="other")
```

| Parameter | Default | Meaning |
|---|---|---|
| `category` | `None` | The document type (usually from classify) — becomes part of every chunk's breadcrumb |
| `vocabulary` | `None` | `{section: description}`, max 50. Presence skips discovery |
| `unmatched` | `None` → `"other"` | What happens to pages matching no category — see below |
| `max_chunk_tokens` | `768` | The chunk-size ceiling |

`unmatched` modes (only meaningful with a `vocabulary`):

- **`other`** (default) — unmatched pages form an honest `other`
  section, still chunked and searchable; `other` joins the result
  vocabulary only when actually produced.
- **`require`** — every page must match: the model gets no escape
  option, and an invalid answer inherits the left neighbor.
- **`skip`** — unmatched pages are dropped before grouping; they are
  still counted in `pages_used` (they were read), and a document where
  everything is skipped returns empty sections.

Unset parameters resolve from `rules.yaml`'s `split:` preset — see
[Content rules](content-rules.md). `vocabulary={}` forces discovery.

## What retrieval sees

Every `Chunk` carries `embedding_text` — the chunk's text prefixed with
its breadcrumb:

```
[research_paper › study_design › patient recruitment]
Patients were recruited across 12 sites...
```

The breadcrumb is contextual retrieval: a chunk about "recruitment"
embedded under its document type and section retrieves far better than
the bare paragraph. Chunks also carry `region_ids` per page (the
provenance trail back to bounding boxes), `kind`, and
`token_estimate`.

Full signatures: [API reference → Operations](../reference/operations.md).
