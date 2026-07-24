# Configuration

ingestlib reads three files, all sitting together in one directory:

```
config.yaml     infrastructure: AWS, providers, stores, servers
rules.yaml      content rules: classification rules + split categories (optional)
.env            secrets: API keys and connection URLs
```

Start from the templates in the repository —
[config.example.yaml](https://github.com/LangModule/ingestlib/blob/main/config.example.yaml),
[rules.example.yaml](https://github.com/LangModule/ingestlib/blob/main/rules.example.yaml),
[.env.example](https://github.com/LangModule/ingestlib/blob/main/.env.example)
(working from a clone, just `cp` each one next to your project):

```bash
cp config.example.yaml config.yaml
cp rules.example.yaml rules.yaml     # optional
cp .env.example .env
```

## How configuration is discovered

Nothing is read at import time — the first library *call* triggers
discovery:

1. `INGESTLIB_CONFIG=/path/to/config.yaml` environment variable wins.
2. Otherwise `config.yaml` is searched for in the working directory,
   then each parent directory.

The `.env` and `rules.yaml` **next to the discovered config.yaml** are
loaded with it. Relative paths inside the config (`artifacts.path`,
`sqlite.path`) anchor to the config file's directory, not the working
directory — the same corpus regardless of where you launch from.

Configuration loads once and is cached. `reset_config()` forgets the
cache, un-sets the secrets the previous `.env` injected, and resets
every client singleton — the next call re-discovers everything. This is
how long-running processes apply config edits without a restart.

```python
from ingestlib.config import get_config, reset_config

get_config().vector_store    # "sqlite"
# ... edit config.yaml ...
reset_config()               # next call reloads
```

## config.yaml reference

The **only required section is `aws`** (even on the OpenAI provider —
it identifies your account for defaults; use placeholder values if you
run fully AWS-free). Every other key has a working default and can be
omitted.

### Top-level choices

| Key | Default | Meaning |
|---|---|---|
| `vector_store` | `pinecone` | Which connector the services use: `sqlite` · `pinecone` · `qdrant` · `pgvector` · `mongodb` · `milvus` · `opensearch` · `weaviate` |
| `reranker` | `jina` | `retrieve()`'s reranker: `jina` · `aws` · `none` |
| `artifact_store` | `s3` | Where pipeline artifacts persist: `s3` · `local` |
| `llm_provider` | `bedrock` | Who serves chat/structured calls: `bedrock` · `openai` |
| `embedding_provider` | `bedrock` | Who embeds text: `bedrock` · `openai` — switching changes the vector space; see [AI providers](../guides/providers.md) |

### `aws` (required)

| Key | Meaning |
|---|---|
| `profile` | Named profile from `~/.aws/credentials` (credentials never live in `.env`) |
| `region` | Bedrock region, e.g. `us-east-1` |
| `account_id` | Your 12-digit account id (used in the default bucket name) |

### `bedrock`

| Key | Default |
|---|---|
| `llm_model_id` | `us.amazon.nova-2-lite-v1:0` |
| `embedding_model_id` | `amazon.nova-2-multimodal-embeddings-v1:0` |
| `rerank_model_id` | `amazon.rerank-v1:0` |
| `rerank_region` | `us-west-2` (the rerank model is not in us-east-1) |

### `openai`

| Key | Default |
|---|---|
| `llm_model_id` | `gpt-5-mini` |
| `embedding_model_id` | `text-embedding-3-small` |

The key itself comes from `OPENAI_API_KEY` in `.env`.

### `jina`

| Key | Default |
|---|---|
| `base_url` | `https://api.jina.ai/v1` |
| `rerank_model_id` | `jina-reranker-v3` |

The key itself comes from `JINA_API_KEY` in `.env`.

### `paddle_vl` (the OCR server)

| Key | Default |
|---|---|
| `backend` | `mlx-vlm-server` (Apple Silicon) — or `vllm-server` (NVIDIA) |
| `server_url` | `http://localhost:8111/` |
| `api_model_name` | `PaddlePaddle/PaddleOCR-VL-1.6` |

### Artifact storage

| Key | Default | Meaning |
|---|---|---|
| `s3.bucket` | `ingestlib-{account_id}` | Created automatically on first use |
| `artifacts.path` | `artifacts` | Local folder (when `artifact_store: local`); relative → beside config.yaml |

### Vector-store sections

Each connector has a small section for names; connection secrets live in
`.env` (table below).

| Section | Keys and defaults |
|---|---|
| `sqlite` | `path: ingestlib.db` (relative → beside config.yaml) |
| `pinecone` | `index_name: ingestlib` · `sparse_index_name: {index_name}-sparse` · `sparse_model_id: pinecone-sparse-english-v0` · `cloud: aws` · `region: us-east-1` |
| `qdrant` | `collection_name: ingestlib` |
| `pgvector` | `table_name: ingestlib` |
| `mongodb` | `database: ingestlib` · `collection_name: ingestlib` |
| `milvus` | `collection_name: ingestlib` |
| `opensearch` | `index_name: ingestlib` |
| `weaviate` | `collection_name: Ingestlib` |

## .env reference

Only the keys your choices require are needed. sqlite needs none.

| Key | Needed when | Notes |
|---|---|---|
| `JINA_API_KEY` | `reranker: jina` | Free at jina.ai |
| `OPENAI_API_KEY` | `llm_provider` or `embedding_provider: openai` | |
| `PINECONE_API_KEY` | `vector_store: pinecone` | |
| `QDRANT_URL` | `vector_store: qdrant` | Default `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant Cloud | Empty for a local server |
| `PGVECTOR_URL` | `vector_store: pgvector` | `postgresql://user:pass@host:5432/db` |
| `MONGODB_URL` | `vector_store: mongodb` | `mongodb+srv://…` or `mongodb://…` |
| `MILVUS_URL` | `vector_store: milvus` | Default `http://localhost:19530` |
| `MILVUS_TOKEN` | Zilliz Cloud | Empty for a local server |
| `OPENSEARCH_URL` | `vector_store: opensearch` | Amazon domains sign with `aws.profile` |
| `WEAVIATE_URL` | `vector_store: weaviate` | Default `http://localhost:8080` |
| `WEAVIATE_API_KEY` | Weaviate Cloud | Empty for a local server |

AWS credentials are **never** in `.env` — the library uses the named
profile from `~/.aws/credentials`.

## rules.yaml reference

The optional content-rules sidecar: what your documents *mean*, kept
separate from infrastructure. Full semantics in
[Content rules](../guides/content-rules.md).

```yaml
classify:
  rules:                      # up to 20 — presence makes classify closed-set
    invoice: Itemized charges, taxes, payment terms
    sec_filing: 10-K/10-Q style financial filings
  target_pages: "1,3,5-7"     # optional 1-based page selection; empty = all
  max_pages: 5                # optional cap after selection; 0 = none

split:
  categories:                 # up to 50 — presence skips vocabulary discovery
    financial_statements: Balance sheets, income statements
    notes: Footnotes and disclosures
  unmatched: other            # pages matching nothing: require | other | skip
```

Per-call arguments always override the file; an absent rules.yaml means
open-ended classification and discovered split vocabulary everywhere.

## Logging

Controlled by environment variables, not the config file:

| Variable | Default | Meaning |
|---|---|---|
| `INGESTLIB_LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |
| `INGESTLIB_LOG_THIRD_PARTY` | off | `1` also shows paddlex/httpx/botocore chatter |
| `INGESTLIB_LOG_COLOR` | on | `0` disables colored output |

Next: [Quickstart](quickstart.md).
