# infra

Copy-paste infrastructure for ingestlib. Nothing here is required to use
the library — sqlite + local artifacts + Ollama need none of it.

## docker-compose.yml

Local servers for the server-backed vector stores and the structured-
retrieval SQL sources, one profile per backend — start exactly the one
you need:

```bash
docker compose -f infra/docker-compose.yml --profile qdrant up -d
docker compose -f infra/docker-compose.yml --profile qdrant down
```

Profiles — INTERNAL: `registry` (ingestlib's own DB, described below). USER
data: `qdrant` | `pgvector` | `mongodb` | `milvus` (three services — the
official standalone shape) | `opensearch` | `weaviate`, plus `mysql` for the
structured-retrieval SQL source; and `all` for contributors running
`make test-all`. Ports and credentials match what `.env.example` documents;
data persists in named volumes (`down -v` wipes it). Every vector-store profile
is verified against its connector's full e2e suite.
Three hard-won details live in the file so you never hit them: pg18 images
changed their volume mount point, mongodb's atlas-local needs both data
AND configdb mounted or restarts crash-loop, and weaviate needs a pinned
CLUSTER_HOSTNAME or a recreated container can't reopen its volume.

`mysql` is the one SQL-source server (not a vector store) — the other
structured-retrieval backends need no container here: sqlite and duckdb
are serverless, and the postgres source reuses the `pgvector` container.

`registry` is the one INTERNAL server — ingestlib's own metadata DB, not a
user data store. Plain Postgres on host port 5433 (5432 is pgvector). ingestlib
reads and writes it as the `ingestlib` owner role (the compose sets
`POSTGRES_USER=ingestlib`). On first boot the container also runs
`registry/init.sql`, which provisions a SELECT-only role `ingestlib_ro` for a
person or tool to query the registry read-only — connect with
`postgresql://ingestlib_ro:ro_pw@localhost:5433/ingestlib`.

`ingestlib registry init` creates the tables (Alembic migrations), on the
bundled Postgres or on your own — wherever `INGESTLIB_REGISTRY_URL` points. It
does NOT create the read-only role; that lives only in `registry/init.sql`. On
your own Postgres, run `registry/init.sql` there once yourself to provision
`ingestlib_ro` (the script is idempotent).

## iam-policy.json

The least-privilege policy the pipeline runs under on the default AWS stack
(Bedrock + S3) — the permission contract, not infra we provision. Everything
else on AWS (your own OpenSearch domain, VPC, etc.) is yours to bring and
secure. Attach it to the IAM user or role whose profile config.yaml names,
after replacing the placeholders:

- `ACCOUNT_ID` — your 12-digit AWS account id
- `BUCKET_NAME` — your artifact bucket (the library default is
  `ingestlib-{account_id}`, matching config.yaml's `s3.bucket` default)

Statements:

- `IngestlibBedrock` covers the Nova LLM and embedding models. The model
  ARNs use a wildcard region because the Nova model id is a cross-region
  inference profile that can route to any of its underlying regions.
- `IngestlibBucket` and `IngestlibObjects` cover the artifact bucket,
  including delete (used by `artifacts.delete_document`, and by the
  lifecycle `remove()` / `sync(..., prune=True)` that build on it).
- `IngestlibRerank` and `IngestlibRerankModel` are needed only when
  config.yaml sets `reranker: aws`. Amazon Rerank is served from
  us-west-2, and `bedrock:Rerank` does not support resource-level
  scoping. Harmless to keep attached while using the jina reranker.

Not using Bedrock (openai/ollama providers) with s3 artifacts? Keep only
the two S3 statements.
