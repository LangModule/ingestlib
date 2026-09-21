# Storage model

ingestlib splits a corpus across **three stores** that share one join key —
the `doc_id` (content checksum): the **registry** holds everything queryable,
the **artifact store** holds the bytes, and the **vector store** holds the
search index. Each is chosen independently.

## The registry — the source of truth

An internal **Postgres** database (`INGESTLIB_REGISTRY_URL`) that records every
document's structure, classification, sections, chunks, extractions, and
lifecycle state. It is authoritative: `get_document(doc_id)` reads a whole
document back from here, the registry-backed retrieve filters (collection,
confidence, kind) run against it, and `verify` audits the corpus against it.
It is ingestlib's own bookkeeping — required for the corpus path, **not** a
user data store. Bring it up with the bundled compose file plus
`ingestlib registry init` (see [Manage a corpus](../how-to/manage-corpus.md)).

## The artifact store — the bytes

A document's **bytes only**, keyed by checksum, on `s3` or a plain `local`
folder: the source file, page renders, figure crops, and the whole-document
markdown — one layout on both backends, documented in
[What just happened](../get-started/first-pipeline.md#whats-in-the-two-stores).

Because the registry keeps the chunks and the artifact store keeps the bytes,
the vector store is *rebuildable*:
[`reindex()`](../how-to/manage-corpus.md#rebuild-the-vector-store-reindex)
re-embeds every document's chunks straight from the registry — no re-parse — so
wiping and rebuilding an index costs embedding time, not pipeline time.

## The vector store — the search index

One vector per chunk, with the chunk's complete payload riding along
(markdown, text, pages, region ids, section, category). Eight connectors
implement one contract:

```python
class VectorStore:
    def upsert_chunks(document_id, chunks, embeddings, category, namespace) -> int
    def query(vector, top_k, filters, namespace, text) -> list[RetrievedChunk]
    def delete_document(document_id, namespace) -> int
```

Guarantees every connector honors:

- **Idempotent upserts** — re-ingesting a document overwrites its vectors
  in place; a re-parse with *fewer* chunks prunes the orphans
- **No infrastructure on the read path** — querying a store that was never
  written to returns `[]`; indexes and collections are created on first
  *upsert* only
- **Namespaces isolate corpora** — same interface on every backend, even
  the ones with no native namespace concept
- **Filters are equality constraints** on `document_id`, `category`,
  `section`, `kind` — unknown fields raise, with the valid list

## Hybrid search

All eight connectors run **two signals** over the same chunks:

- **Dense** — cosine similarity over the embeddings
- **Lexical** — BM25/full-text over the chunk text, with the breadcrumb
  boosted over the body

When retrieve() passes the question text, both run and the results fuse
(Reciprocal Rank Fusion, server-side where the backend supports it). The
lexical side is what catches exact tokens — invoice numbers, names, error
codes — that embedding similarity smooths over. A lexical failure always
degrades to dense-only with a warning, never an error.

The reranker then produces the final order across both signals' candidates.

## Choosing a backend

| `vector_store` | Runs on | Best when |
|---|---|---|
| `sqlite` (default) | one local file — sqlite-vec + FTS5 | zero setup, single machine, honest hybrid search |
| `pgvector` | the Postgres you already run | you want vectors next to your app's data |
| `qdrant` | local docker or Qdrant Cloud | a dedicated vector DB, server-side fusion |
| `pinecone` | serverless cloud | managed, nothing to operate |
| `mongodb` | Atlas / mongodb 8.2+ | your data already lives in Mongo |
| `milvus` | local docker or Zilliz Cloud | large-scale dedicated vector infra |
| `opensearch` | Amazon domain or docker | AWS-native search stack |
| `weaviate` | local docker or Weaviate Cloud | server-side hybrid in one call |

Connection details per backend are in
[Connect a vector store](../how-to/vector-stores.md). Only the selected
connector ever builds a client — the others' SDKs are never imported, and
(except sqlite, which ships with the core install) not even installed
unless you add their [pip extra](../get-started/installation.md#1-install-the-package).

---

That's the last concept. From here, the [how-to guides](../how-to/parse-documents.md)
cover the tasks.
