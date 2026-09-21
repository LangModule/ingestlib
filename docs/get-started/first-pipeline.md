# What just happened

The quickstart's two calls — `ingest()` and `retrieve()` — produced a lot
more than two return values. This page walks through all of it, because the
artifacts are the mental model for everything else in these docs.

## The five stages

`ingest("report.pdf")` ran this pipeline:

```mermaid
flowchart LR
  P[parse] --> C[classify] --> S[split] --> E[embed] --> U[upsert]
  P -. "source · page images" .-> ART[(artifact store)]
  C -. "structure · labels · chunks" .-> REG[(registry)]
  S -. .-> REG
  U -. "one vector per chunk" .-> VEC[(vector store)]
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

## What's in the two stores

Every stage's output was persisted under `doc_id`, split across the two stores
it belongs in. Queryable structure — pages, regions, classification, sections,
chunks, extractions — lands in the **registry** (Postgres). The **bytes** land
in the artifact store; with `artifact_store: local` you can browse them in a
file manager:

```text
artifacts/documents/{doc_id}/
├── source/report.pdf                 the original bytes
├── parse/document.md                 whole-document markdown
├── parse/pages/page_0001.png …       full page renders
└── parse/figures/…png                cropped charts & figures
```

Read any of it back without re-running anything:

```python
from ingestlib.services import get_document

doc = get_document(doc_id)          # structure from the registry
doc.category, doc.chunk_count       # queryable metadata
doc.chunks                          # the stored chunks, with provenance
doc.markdown()                      # whole-document markdown (from the artifact store)
doc.page_image(1)                   # a page render (bytes)
```

Dedup means the pipeline never re-runs for an unchanged file — and your own
tooling can `get_document` the stored result instead of ever parsing again.

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

Those `region_ids` point back into the registry's stored bounding boxes —
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
