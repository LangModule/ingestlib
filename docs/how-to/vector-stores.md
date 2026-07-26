# Connect a vector store

Three steps for any backend: install its extra (sqlite needs none — it
ships with the core install), pick one key in `config.yaml`, put its
connection secret in `.env`. Indexes, collections, tables, and search
schemas are all created automatically on first ingest. All eight
connectors deliver [hybrid search](../concepts/storage.md#hybrid-search);
switching stores never changes your code.

```yaml
vector_store: sqlite   # or any tab below
```

If you select a store whose SDK isn't installed, the error names the exact
extra: `pip install "ingestlib[qdrant]"`.

=== "sqlite"

    The default. One local file, no server, no keys — sqlite-vec for
    dense KNN, FTS5 BM25 for lexical, fused with RRF.

    ```yaml
    vector_store: sqlite
    # sqlite:
    #   path: ingestlib.db      # relative paths anchor beside config.yaml
    ```

    Nothing to install or run. Best single-machine choice.

=== "Pinecone"

    Serverless, fully managed. Two indexes (dense + hosted sparse model)
    are created on first use.

    ```bash
    uv add "ingestlib[pinecone]"
    ```

    ```yaml
    vector_store: pinecone
    ```

    ```bash
    # .env — key from https://app.pinecone.io → API Keys
    PINECONE_API_KEY=…
    ```

=== "Qdrant"

    Local docker or Qdrant Cloud. Dense + BM25 sparse on one collection,
    fused server-side.

    ```bash
    uv add "ingestlib[qdrant]"
    ```

    ```yaml
    vector_store: qdrant
    ```

    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

    ```bash
    # .env
    QDRANT_URL=http://localhost:6333
    QDRANT_API_KEY=            # cloud only
    ```

=== "pgvector"

    The Postgres you already run — HNSW cosine + built-in full-text over a
    generated tsvector. The extension is enabled and the table created
    automatically.

    ```bash
    uv add "ingestlib[pgvector]"
    ```

    ```yaml
    vector_store: pgvector
    ```

    ```bash
    docker run -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector:pg18
    ```

    ```bash
    # .env
    PGVECTOR_URL=postgresql://postgres:pw@localhost:5432/postgres
    ```

    Works with RDS, Supabase, Neon — anywhere the extension ships.

=== "MongoDB"

    Atlas Vector Search + Atlas Search (true BM25). Atlas any tier, the
    atlas-local docker image, or self-managed 8.2+ with mongot.

    ```bash
    uv add "ingestlib[mongodb]"
    ```

    ```yaml
    vector_store: mongodb
    ```

    ```bash
    docker run -p 27017:27017 mongodb/mongodb-atlas-local
    ```

    ```bash
    # .env
    MONGODB_URL=mongodb://localhost:27017/?directConnection=true
    ```

    Search indexes sync a few seconds behind writes — ingestion is
    unaffected; only write-then-query-immediately notices.

=== "Milvus"

    Dense ANN + server-computed BM25, fused server-side. Local standalone
    docker or Zilliz Cloud.

    ```bash
    uv add "ingestlib[milvus]"
    ```

    ```yaml
    vector_store: milvus
    ```

    ```bash
    # .env
    MILVUS_URL=http://localhost:19530
    MILVUS_TOKEN=              # Zilliz Cloud only
    ```

=== "OpenSearch"

    faiss k-NN + Lucene BM25. An Amazon OpenSearch domain — requests are
    SigV4-signed with your configured `aws.profile`, no separate key — or
    a local server.

    ```bash
    uv add "ingestlib[opensearch]"
    ```

    ```yaml
    vector_store: opensearch
    ```

    ```bash
    docker run -p 9200:9200 -e discovery.type=single-node \
      -e DISABLE_SECURITY_PLUGIN=true opensearchproject/opensearch
    ```

    ```bash
    # .env
    OPENSEARCH_URL=http://localhost:9200
    ```

=== "Weaviate"

    HNSW dense + native BM25, fused server-side in one hybrid call. Local
    docker (publish **both** ports — the client speaks gRPC too) or
    Weaviate Cloud.

    ```bash
    uv add "ingestlib[weaviate]"
    ```

    ```yaml
    vector_store: weaviate
    ```

    ```bash
    docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
    ```

    ```bash
    # .env
    WEAVIATE_URL=http://localhost:8080
    WEAVIATE_API_KEY=          # cloud only
    ```

## Verify

```bash
uv run ingestlib doctor        # includes a reachability check for the selected store
```

## Behaviors shared by every connector

- **Created on first use** — no manual index/collection/table setup, and
  nothing is created on the read path
- **Idempotent re-ingestion** — same document overwrites in place; a
  re-parse with fewer chunks prunes the leftovers
- **Namespaces** and **filters** work identically everywhere
- Index/collection/table **names are configurable** per backend — see the
  [configuration reference](../reference/configuration.md)
- Only the **selected** connector's SDK is ever used at runtime

## Migrating between stores

Artifacts are the source of truth, so moving stores is a re-ingest, not an
export: change `vector_store`, then re-run ingestion over your corpus —
parses are reused from the artifact store, so no OCR or LLM calls repeat.
