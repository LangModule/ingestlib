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

## Deduplication and versioning

Documents have two identities: content (`doc_id = sha256(bytes)`) and logical
(`namespace` + source path). By default (`skip_existing=True`) ingest reconciles
against both:

| status | when |
|---|---|
| `skipped` | same bytes, same path, already fully ingested |
| `moved` | same bytes, new path — only the registry is re-pointed |
| `replaced` | this path held an **older** version; it's deleted after the new one goes live |
| `ingested` | new document |

- A run that **failed partway is retried** — only a completed pipeline (manifest
  written) counts as done.
- **Editing a file and re-ingesting replaces the old version** — vectors and
  artifacts of the previous version are removed, so retrieval never returns
  stale content. This is automatic; the [corpus guide](manage-corpus.md) covers
  it in full.

```python
r = ingest("report.pdf")                        # edited file → status="replaced"
r.replaced_doc_id                               # the old version, now deleted

ingest("report.pdf", skip_existing=False)       # force a re-run of unchanged bytes
ingest("v2.pdf", replaces=old_doc_id)           # supersede a version at another path
```

Re-ingesting the same content overwrites its vectors in place — never
duplicates, and stale chunks from a previous run are pruned.

To reconcile a whole folder (add/replace/move/prune) in one call, use
[`sync()`](manage-corpus.md#sync-a-folder).

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

One call erases it from both stores (vectors first, then artifacts):

```python
from ingestlib.services import remove

remove("report.pdf")     # by source path, or a doc_id / unique prefix
```

See [Manage a corpus](manage-corpus.md#remove-one-document) for the full
lifecycle — replace, sync, prune, backfill.

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
