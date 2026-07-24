# Ingest

Commit a document to your stack: the full pipeline, every stage
persisted, ending with searchable vectors.

```python
from ingestlib.services import ingest

r = ingest("report.pdf")
print(r.status)      # "ingested" | "skipped"
print(r.doc_id)      # content hash — the document's permanent id
print(r.category, r.sections, r.chunks, r.vectors)
print(r.durations)   # {'parse': 26.3, 'classify': 2.1, 'split': 8.4, ...}
```

Async: `aingest`.

## The five stages

```
parse → save   classify → save   split → save   embed   upsert → manifest
```

Each result is written to the [artifact store](artifact-store.md)
immediately after its stage; embeddings are computed from each chunk's
`embedding_text` (in parallel) and upserted into the configured
[vector store](vector-stores.md). The **ingest manifest is written
last** — that ordering is what makes dedup honest.

## Dedup and resume

`doc_id` is the SHA-256 of the file's bytes. With `skip_existing=True`
(the default), a document whose **manifest** exists is skipped:

- The same file ingested twice → the second call returns
  `status="skipped"` instantly, nothing recomputed.
- A run that died partway (parsed and classified, then crashed) has no
  manifest → the next ingest **runs again** rather than lying about
  completeness. Nothing is ever half-ingested silently.

One byte of difference is a different `doc_id` — content addressing has
no notion of "new version of the same document".

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `store` | `None` → config's `vector_store` | Any `VectorStore` instance — bring your own connector or configuration |
| `namespace` | `""` | Partition within the store; queries only see their own namespace |
| `skip_existing` | `True` | Content-hash dedup (manifest-based, see above) |
| `max_chunk_tokens` | `768` | Passed through to split |
| `on_stage` | `None` | Progress callback — see below |

The classify and split stages honor `rules.yaml` presets automatically
([content rules](content-rules.md)).

## Live progress: `on_stage`

For UIs and long documents:

```python
def on_stage(stage: str, event: str) -> None:   # event: "start" | "done"
    print(f"{stage}: {event}")

ingest("big.pdf", on_stage=on_stage)
```

Contract details worth knowing:

- Exceptions in your callback are logged and swallowed — a broken
  progress bar can never kill an ingest.
- A stage that fails leaves its `"start"` without a `"done"` — that's
  your failure attribution.
- The dedup fast path emits nothing (nothing ran).

This is exactly the hook the studio streams over SSE for its live
five-stage stepper.

## What you get back

`IngestResult`: `status`, `doc_id`, `category`, `confidence`, `pages`,
`sections`, `chunks`, `vectors`, and per-stage `durations`.

Full signatures: [API reference → Services](../reference/services.md).
