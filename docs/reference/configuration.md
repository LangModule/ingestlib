# Configuration reference

Every key, every default. Anything omitted from `config.yaml` uses the
default shown here — the file only needs your choices
([the model](../concepts/configuration.md)).

## Discovery

1. `INGESTLIB_CONFIG=/path/to/config.yaml` environment variable
2. Otherwise: the working directory, then its parents

`.env` and `rules.yaml` are read from beside the discovered `config.yaml`.

## Top-level choices

| Key | Default | Options |
|---|---|---|
| `llm_provider` | `bedrock` | `bedrock` · `openai` · `ollama` |
| `embedding_provider` | `bedrock` | `bedrock` · `openai` · `ollama` |
| `vector_store` | `sqlite` | `sqlite` · `pinecone` · `qdrant` · `pgvector` · `mongodb` · `milvus` · `opensearch` · `weaviate` |
| `artifact_store` | `s3` | `s3` · `local` |
| `reranker` | `jina` | `jina` · `aws` · `none` |

## `aws` — conditional

Required only while a choice uses AWS (bedrock provider, s3 artifacts, aws
reranker, Amazon OpenSearch domain). No defaults — all three keys required
when the section exists.

| Key | Meaning |
|---|---|
| `aws.profile` | profile from `~/.aws/credentials`; empty string = default credential chain |
| `aws.region` | e.g. `us-east-1` |
| `aws.account_id` | quoted string; used for the default bucket name |

## AI providers

| Key | Default |
|---|---|
| `bedrock.llm_model_id` | `us.amazon.nova-2-lite-v1:0` |
| `bedrock.embedding_model_id` | `amazon.nova-2-multimodal-embeddings-v1:0` |
| `bedrock.rerank_model_id` | `amazon.rerank-v1:0` |
| `bedrock.rerank_region` | `us-west-2` |
| `openai.llm_model_id` | `gpt-5-mini` |
| `openai.embedding_model_id` | `text-embedding-3-small` |
| `ollama.base_url` | `http://localhost:11434/v1` — any OpenAI-compatible server |
| `ollama.llm_model_id` | `qwen3.5:9b` |
| `ollama.embedding_model_id` | `qwen3-embedding:0.6b` |
| `jina.base_url` | `https://api.jina.ai/v1` |
| `jina.rerank_model_id` | `jina-reranker-v3` |

## OCR

| Key | Default |
|---|---|
| `paddle_vl.backend` | `mlx-vlm-server` — or `vllm-server` (NVIDIA) |
| `paddle_vl.server_url` | `http://localhost:8111/` |
| `paddle_vl.api_model_name` | `PaddlePaddle/PaddleOCR-VL-1.6` |

## Artifact stores

| Key | Default |
|---|---|
| `s3.bucket` | `ingestlib-{aws.account_id}` — names are global across AWS |
| `artifacts.path` | `artifacts` — local mode's folder; relative paths anchor beside config.yaml |

## Vector stores

Names are created on first use; only the selected backend's keys are read.
Server-backed stores also need their
[pip extra](../get-started/installation.md#1-install-the-package) —
sqlite ships with the core install.

| Key | Default |
|---|---|
| `sqlite.path` | `ingestlib.db` — relative anchors beside config.yaml |
| `pinecone.index_name` | `ingestlib` |
| `pinecone.sparse_index_name` | `ingestlib-sparse` |
| `pinecone.sparse_model_id` | `pinecone-sparse-english-v0` |
| `pinecone.cloud` / `pinecone.region` | `aws` / `us-east-1` |
| `qdrant.collection_name` | `ingestlib` |
| `pgvector.table_name` | `ingestlib` |
| `mongodb.database` / `mongodb.collection_name` | `ingestlib` / `ingestlib` |
| `milvus.collection_name` | `ingestlib` |
| `opensearch.index_name` | `ingestlib` |
| `weaviate.collection_name` | `Ingestlib` — Weaviate capitalizes collection names |

## Environment variables (`.env`)

Secrets never live in config.yaml. Only the selected backends' variables
are read.

| Variable | Needed when |
|---|---|
| `JINA_API_KEY` | `reranker: jina` |
| `OPENAI_API_KEY` | `llm_provider` or `embedding_provider: openai` |
| `PINECONE_API_KEY` | `vector_store: pinecone` |
| `QDRANT_URL` · `QDRANT_API_KEY` | `vector_store: qdrant` (key: cloud only) |
| `PGVECTOR_URL` | `vector_store: pgvector` — `postgresql://user:pw@host:5432/db` |
| `MONGODB_URL` | `vector_store: mongodb` |
| `MILVUS_URL` · `MILVUS_TOKEN` | `vector_store: milvus` (token: Zilliz Cloud) |
| `OPENSEARCH_URL` | `vector_store: opensearch` — Amazon domains SigV4-sign via `aws.profile` |
| `WEAVIATE_URL` · `WEAVIATE_API_KEY` | `vector_store: weaviate` (key: cloud only) |

Non-secret environment controls:

| Variable | Effect |
|---|---|
| `INGESTLIB_CONFIG` | explicit config.yaml path — wins over discovery |
| `INGESTLIB_LOG_LEVEL` | `DEBUG` · `INFO` (default) · `WARNING` · `ERROR` |
| `INGESTLIB_LOG_THIRD_PARTY` | `1` raises SDK loggers to the same level |
| `INGESTLIB_LOG_COLOR` | `0` disables colored output |

## `rules.yaml` — content rules (optional)

Lives beside config.yaml; used whenever a call passes no explicit rules
([guide](../how-to/content-rules.md)).

```yaml
classify:
  max_pages: 5                # optional page cap
  target_pages: "1,3,5-7"     # optional 1-based selection
  rules:                      # up to 20 {label: description}
    invoice: "Itemized charges, tax info, and payment terms"

split:
  unmatched: other            # other | require | skip
  categories:                 # up to 50 {section: description}
    financial_statements: "Balance sheets, income statements, cash flows"
```
