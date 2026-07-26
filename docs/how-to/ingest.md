# Ingest documents

`ingest()` is the whole pipeline in one call: parse → classify → split →
embed → upsert, with every stage persisted. This page covers the behaviors
you control.

## The basic call

```python
from ingestlib.services import ingest

r = ingest("report.pdf")
print(r.status, r.category, r.chunks, r.vectors)
print(r.durations)     # per-stage seconds — parse dominates
```

Accepts PDF, DOCX, PPTX, and PNG/JPEG/WebP images.

## Deduplication

Documents are identified by content checksum. By default
(`skip_existing=True`):

- the same bytes ingested again → `status="skipped"`, nothing runs
- a run that **failed partway is retried** — only a fully completed
  pipeline (its ingest manifest written) counts as done
- dedup keys on content **only** — different rules or settings still skip;
  force a re-run with `skip_existing=False`

```python
r = ingest("report.pdf", skip_existing=False)   # re-run regardless
```

Re-ingestion overwrites the document's vectors in place — never
duplicates, and stale chunks from a previous run are pruned.

## Progress reporting

Long documents take minutes (parse dominates). `on_stage` reports each
stage's lifecycle — feed a progress bar or a job log:

```python
def on_stage(stage: str, event: str) -> None:
    print(f"{stage}: {event}")     # parse: start … parse: done … classify: start …

ingest("big.pdf", on_stage=on_stage)
```

Stages are `parse | classify | split | embed | upsert`; events are
`start | done`. Exceptions your callback raises are logged and ignored —
a broken progress bar never kills an ingest.

## Content rules per call

All the [content-rule arguments](content-rules.md) pass straight through:

```python
ingest("doc.pdf",
    categories={"invoice": "…"},      # classify rules
    target_pages="1-3", max_pages=3,  # classify page selection
    vocabulary={"line_items": "…"},   # split sections
    unmatched="skip",                 # drop everything else
)
```

## Namespaces

Isolate corpora inside one vector store — per tenant, per project, per
embedding model:

```python
ingest("doc.pdf", namespace="tenant-a")
retrieve("question", namespace="tenant-a")   # only tenant-a's documents
```

## Chunk size

```python
ingest("doc.pdf", max_chunk_tokens=512)      # default 768
```

Boundaries still follow content — this is a ceiling, not a target; tables
and figures are never split regardless.

## Delete a document

Remove it from both stores:

```python
from ingestlib.storage import artifacts, default_store

store = default_store()
store.delete_document(doc_id)        # vectors gone
artifacts.delete_document(doc_id)    # parse/classify/split artifacts gone
```

## A folder at a time

```python
from pathlib import Path

for path in sorted(Path("corpus/").glob("*.pdf")):
    r = ingest(path)
    print(r.status, path.name)
```

Dedup makes this idempotent — re-running the loop only processes new
files.

---

Next: [Retrieve & filter](retrieve.md).
