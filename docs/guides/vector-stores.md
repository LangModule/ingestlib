# Vector stores

Eight connectors, one contract. Every one supports **hybrid search**
(dense + lexical), metadata filtering, namespaces, and idempotent
upsert/delete. Pick with one key:

```yaml
# config.yaml
vector_store: sqlite    # or pinecone | qdrant | pgvector | mongodb
                        #    | milvus | opensearch | weaviate
```

All indexes, collections, and tables **bootstrap on first use** — no
manual setup. In our retrieval evals, every connector scores
identically (hit@3 = 1.00 reranked), so choose on operational grounds:

| Pick | When |
|---|---|
| **SQLite** | You want zero infrastructure: one local file, no server, no keys. Retrieves as well as the clouds in our evals — the right default for local and single-machine use |
| **Pinecone** | You want serverless cloud with a free tier and nothing to operate |
| **Qdrant** | You want an OSS vector DB — local docker or Qdrant Cloud |
| **pgvector** | You already run Postgres (RDS, Supabase, Neon, docker) |
| **MongoDB** | You already run MongoDB (Atlas any tier incl. free M0, or 8.2+ self-managed) |
| **Milvus** | You want a dedicated OSS vector database at scale — docker or Zilliz Cloud |
| **OpenSearch** | You are AWS-native — Amazon domains sign with your existing profile, no new key |
| **Weaviate** | You want native server-side hybrid in a single call — docker or Weaviate Cloud |

## Using a connector directly

The services use the configured store automatically. To bring your own
instance (different collection, several stores side by side):

```python
from ingestlib.storage import SqliteStore, default_store
from ingestlib.services import ingest, retrieve

store = default_store()          # whatever config.yaml selects
ingest("report.pdf", store=SqliteStore())   # or any connector instance
retrieve("question", store=store)
```

The contract every connector implements is the `VectorStore` ABC:
`upsert_chunks`, `query` (dense vector + optional query text for the
lexical half), `delete_document` — see
[API reference → Storage](../reference/storage.md).

---

## Connector notes

What each needs and how its hybrid works. Config keys (index/collection
names) are in the [configuration reference](../getting-started/configuration.md);
everything below defaults sensibly.

### SQLite

- **Needs**: nothing. `sqlite.path` (default `ingestlib.db`, beside
  config.yaml).
- **Hybrid**: sqlite-vec (`vec0`) dense KNN + built-in FTS5 BM25 with
  porter stemming (breadcrumb weighted 2×), fused client-side with RRF.
- **Notes**: writes are one transaction across all tables — dense and
  lexical can never drift. Namespaces are physical partitions.

### Pinecone

- **Needs**: `PINECONE_API_KEY`.
- **Hybrid**: two serverless indexes — dense (cosine) + sparse, with
  Pinecone-hosted sparse embeddings (`pinecone-sparse-english-v0`, no
  corpus state to manage); merged client-side, reranker arbitrates.

### Qdrant

- **Needs**: `QDRANT_URL` (+ `QDRANT_API_KEY` for cloud).
  Local: `docker run -p 6333:6333 qdrant/qdrant`
- **Hybrid**: one collection with named dense + BM25 sparse vectors
  (server computes IDF live); both branches fused **server-side** with
  RRF in a single query.

### pgvector

- **Needs**: `PGVECTOR_URL` (`postgresql://user:pass@host:5432/db`).
  Local: `docker run -p 5432:5432 pgvector/pgvector:pg18`
- **Hybrid**: HNSW cosine + a generated, weighted `tsvector` column
  (lexical index physically cannot drift from the text), fused
  client-side with RRF.
- **Notes**: the `vector` extension is enabled automatically when the
  role may; otherwise you get a clear message naming the fix. Writes
  are delete-then-insert in one transaction — re-ingests never leave
  orphaned chunks.

### MongoDB

- **Needs**: `MONGODB_URL` (Atlas `mongodb+srv://…` or self-managed
  8.2+).
- **Hybrid**: `$vectorSearch` (cosine) + `$search` (true Lucene BM25,
  breadcrumb boosted), fused client-side with RRF. Both search indexes
  are created programmatically and polled until queryable.
- **Notes**: Atlas search indexes lag writes by seconds (eventual
  consistency). The free M0 tier caps search indexes at 3 per cluster.

### Milvus

- **Needs**: `MILVUS_URL` (+ `MILVUS_TOKEN` for Zilliz Cloud).
- **Hybrid**: dense + a built-in BM25 function (the server computes
  sparse vectors and IDF from raw text), fused **server-side** with an
  RRF ranker in one `hybrid_search` call.

### OpenSearch

- **Needs**: `OPENSEARCH_URL`. Amazon domains are SigV4-signed with
  your `aws.profile` — no new credential; local docker runs unsigned.
- **Hybrid**: faiss HNSW k-NN + Lucene BM25, fused client-side with
  RRF.

### Weaviate

- **Needs**: `WEAVIATE_URL` (+ `WEAVIATE_API_KEY` for cloud). Local
  docker needs HTTP + gRPC ports (8080, 50051).
- **Hybrid**: native — one `hybrid` call fuses dense and BM25
  server-side.

---

## Migrating between stores

The vector store is a **derived index**; the artifact store is the
source of truth. Switching stores is not a migration project:

1. Change `vector_store` in config.yaml.
2. Re-embed the stored chunks into the new store — no re-parsing, no
   OCR server (the studio's Settings page does this as "Backfill", or
   see the eval harness's `--backfill` flag).

The previous store is left untouched, so two stores can stay populated
side by side.
