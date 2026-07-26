# What just happened

The quickstart's two calls — `ingest()` and `retrieve()` — produced a lot
more than two return values. This page walks through all of it, because the
artifacts are the mental model for everything else in these docs.

## The five stages

`ingest("report.pdf")` ran this pipeline:

```text
parse ──▶ classify ──▶ split ──▶ embed ──▶ upsert
  │           │           │                   │
  ▼           ▼           ▼                   ▼
        the artifact store              the vector store
```

The `IngestResult` you got back summarizes it:

```python
r = ingest("report.pdf")
r.doc_id      # '7b6b95d79149…' — sha-256 of the file's bytes
r.category    # 'research_paper' — what classify decided
r.sections    # 6  — page groups split discovered
r.chunks      # 24 — retrieval units created
r.vectors     # 24 — embeddings upserted
r.durations   # {'parse': 41.2, 'classify': 3.1, 'split': 9.8, ...}
```

## The document's identity

`doc_id` is the SHA-256 checksum of the file's content. That single fact
drives several behaviors:

- **Dedup** — ingesting the same bytes again returns `status="skipped"`
  without running anything. A run that failed partway *is* retried: only a
  fully completed pipeline counts.
- **Overwrite-in-place** — re-ingesting after `skip_existing=False`
  replaces the document's vectors, never duplicates them.
- **A one-byte change is a new document** — content addressing has no
  notion of "version 2 of the same file".

## What's in the artifact store

Every stage's output was persisted, keyed by `doc_id`. With
`artifact_store: local` you can browse it in a file manager:

```text
artifacts/documents/{doc_id}/
├── source/report.pdf                 the original bytes
├── parse/result.json                 every region, bbox, and markdown block
├── parse/document.md                 whole-document markdown
├── parse/pages/page_0001.png …       full page renders
├── parse/figures/…png                cropped charts & figures
├── classify/result.json              category + confidence + reasoning
├── split/result.json                 sections and chunks, with provenance
├── split/ingest_manifest.json        what went to the vector store, when
└── meta.json                         registry entry (filename, pages, counts)
```

Load any of it back without re-running anything:

```python
from ingestlib.storage import artifacts

artifacts.list_documents()               # registry of everything stored
parse   = artifacts.load_parse(doc_id)   # full ParseResult (structure only)
chunks  = artifacts.load_split(doc_id).chunks
```

This is why parse never runs twice for the same file: classify, split, and
any of your own tooling reuse the stored result.

## What's in the vector store

One vector per chunk. Each vector carries the chunk's full payload —
markdown, text, pages, region ids, section, category — so a query hit
needs no second lookup to be useful:

```python
result = retrieve("how were participants recruited?")
hit = result.hits[0]

hit.chunk.markdown     # the chunk's content
hit.chunk.pages        # [4]
hit.chunk.region_ids   # {4: [2, 3]} — page → region ids on that page
hit.citation           # 'doc 7b6b95d79149 · p.4 · methods'
```

Those `region_ids` point back into `parse/result.json`'s bounding boxes —
which is how an answer becomes a highlight on the original page. That
chain is the subject of [Provenance & citations](../concepts/provenance.md).

## The embedding text

Chunks aren't embedded raw. Each one is embedded as its
`embedding_text` — the content prefixed with a breadcrumb of where it
lives:

```text
[research_paper › methods › Participant recruitment]

Participants were recruited through community centers in Cairo…
```

That context line is why a query like "recruitment methodology" finds the
right chunk even when the chunk's own words never say "methodology".

---

You now know what every stage produces and where it lives. Continue into
[Concepts](../concepts/pipeline.md) for the deeper model, or jump straight
to the [how-to guides](../how-to/parse-documents.md).
