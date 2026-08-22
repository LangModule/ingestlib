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

Profiles: `qdrant` | `pgvector` | `mongodb` | `milvus` (three services —
the official standalone shape) | `opensearch` | `weaviate`, plus `mysql`
for the structured-retrieval SQL source, and `all` for contributors
running `make test-all`. Ports and credentials match what `.env.example`
documents; data persists in named volumes (`down -v` wipes it). Every
vector-store profile is verified against its connector's full e2e suite.
Three hard-won details live in the file so you never hit them: pg18 images
changed their volume mount point, mongodb's atlas-local needs both data
AND configdb mounted or restarts crash-loop, and weaviate needs a pinned
CLUSTER_HOSTNAME or a recreated container can't reopen its volume.

`mysql` is the one SQL-source server (not a vector store) — the other
structured-retrieval backends need no container here: sqlite and duckdb
are serverless, and the postgres source reuses the `pgvector` container.

The AWS files below cover the managed side. In the two JSON policies,
replace the placeholders before attaching:

- `ACCOUNT_ID` — your 12-digit AWS account id
- `BUCKET_NAME` — your artifact bucket (the library default is
  `ingestlib-{account_id}`, matching config.yaml's `s3.bucket` default)

opensearch.yaml needs no editing — it takes its values as CloudFormation
parameters.

## iam-policy.json

The least-privilege policy the pipeline runs under. Attach it to the IAM
user or role whose profile config.yaml names.

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

## opensearch.yaml

CloudFormation template for the cheapest k-NN-capable OpenSearch domain —
the quickest way to run the OpenSearch connector against a real managed
domain. One r8g.medium.search data node, 1-AZ, no standby, 10 GiB gp3,
public endpoint with fine-grained access control mapped to the IAM
principal you pass as `MasterUserArn`. Roughly $0.10 per hour while it
exists. Deploy, endpoint, and delete commands are in the template header.

There is no stop/start for OpenSearch domains, so delete the stack
whenever work pauses and recreate it when you resume — re-ingesting your
corpus rebuilds the index in the fresh domain.

## iam-deploy-policy.json

Extra permissions for the IAM user that deploys opensearch.yaml:
CloudFormation control scoped to the ingestlib-opensearch stack, the two
template operations that take no resource scoping, and OpenSearch domain
administration scoped to domains named `ingestlib*`. The pipeline itself
never needs these — attach them only to run the stack commands.
