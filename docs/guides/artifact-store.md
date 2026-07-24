# Artifact store

Every operation's output persists to the artifact store — the
**source of truth** of your corpus. The vector store is just a derived,
rebuildable index over it.

```yaml
# config.yaml
artifact_store: s3       # durable, shareable — bucket created on first use
# or
artifact_store: local    # a plain folder beside config.yaml — zero cloud
```

Both backends use the identical layout, so moving a corpus between them
is a copy:

```
documents/{doc_id}/
├── source/{filename}                original file, exact bytes
├── parse/result.json                ParseResult (image bytes stripped)
├── parse/document.md                whole-document markdown
├── parse/pages/page_0001.png …      page renders
├── parse/figures/…                  figure/chart crops
├── classify/result.json             ClassifyResult
├── split/result.json                SplitResult (chunks with provenance)
└── split/ingest_manifest.json       vector-store sync record
```

`doc_id` is the parse checksum: re-saving the same file overwrites in
place, and "already ingested?" is a single existence check. The
citation chain needs no database — a retrieval hit's
`{doc_id, pages, region_ids}` resolves to page images and bounding
boxes straight from this layout.

## Why this design matters

- **Nothing is ever computed twice.** A saved parse can be
  re-classified, re-split with new rules, or re-embedded into a new
  vector store — without touching the OCR server.
- **Vector stores are disposable.** Switch connectors or embedding
  providers, then rebuild the index from stored splits
  ([backfill](vector-stores.md#migrating-between-stores)).
- **UIs need no schema.** The studio's entire library view reads this
  layout directly.

## The API

`ingest()` writes everything automatically. For building on top:

```python
from ingestlib.storage import artifacts
from ingestlib.operations import parse

doc_id = artifacts.save_parse(parse("report.pdf"))

meta = artifacts.list_documents()               # the corpus registry
result = artifacts.load_parse(doc_id)           # structure only (cheap)
result = artifacts.load_parse(doc_id, include_images=True)   # + page PNGs

artifacts.document_exists(doc_id)               # dedup check
png = artifacts.read_blob(f"documents/{doc_id}/parse/pages/page_0001.png")
artifacts.delete_document(doc_id)               # everything under the prefix
```

`save_classify` / `load_classify`, `save_split` / `load_split`, and
`save_ingest_manifest` / `load_ingest_manifest` follow the same
pattern. `read_blob` is the backend-agnostic way for a UI to serve
images when a presigned URL is not available (`artifact_store: local`).

Full signatures: [API reference → Storage](../reference/storage.md).

## Choosing a backend

| | `s3` | `local` |
|---|---|---|
| Durability & sharing | Bucket policies, versioning, team access | Your disk |
| Setup | AWS account; bucket auto-created (`ingestlib-{account_id}`) | Nothing — `artifacts/` beside config.yaml |
| Page-image serving | Presigned URLs | `read_blob` bytes |
| Fits | Teams, production | Local work, zero-cloud setups |

Changing the backend points the library at a different registry —
existing documents stay where they were written until you copy them.
