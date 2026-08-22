# Configuration model

Three files with a strict separation of concerns, all discovered at call
time — never at import.

| File | Holds | Secret? |
|---|---|---|
| `config.yaml` | **Choices** — providers, stores, model ids | No |
| `.env` | **Secrets** — API keys, connection URLs | Yes — never commit |
| `rules.yaml` | **Content rules** — your categories & section vocabulary | No; optional |
| `sources.yaml` | **Structured-retrieval sources** — SQL databases & corpora to query | No; optional |

`ingestlib init` scaffolds the first two; `rules.yaml` and `sources.yaml` are
yours to add when you want [your own categories](../how-to/content-rules.md) or
[query databases](../how-to/structured-retrieval.md). Both sidecars are read
from beside the discovered `config.yaml`, and both use `.env` for any secrets.

## Discovery

Configuration is located when the first call needs it:

1. `INGESTLIB_CONFIG=/path/to/config.yaml` — explicit, wins always
2. Otherwise the working directory is searched, then its parents

`.env`, `rules.yaml`, and `sources.yaml` are read from beside the discovered
`config.yaml`. This is why `ingestlib init` writes into your current directory,
and why a library import never fails on a missing config — only a call can.

## Only your choices live in the file

Every key has a working default; `config.yaml` states what you *chose*.
A fully valid zero-cloud config is five lines:

```yaml
llm_provider: ollama
embedding_provider: ollama
vector_store: sqlite
artifact_store: local
reranker: none
```

The top-level choices:

| Key | Options (default first) | What it selects |
|---|---|---|
| `llm_provider` | `bedrock` · `openai` · `ollama` | who serves chat/vision/structured calls |
| `embedding_provider` | `bedrock` · `openai` · `ollama` | who embeds chunks and queries |
| `vector_store` | `sqlite` · `pinecone` · `qdrant` · `pgvector` · `mongodb` · `milvus` · `opensearch` · `weaviate` | where vectors live |
| `artifact_store` | `s3` · `local` | where stage outputs live |
| `reranker` | `jina` · `aws` · `none` | retrieve()'s second-stage ranker |

Per-backend sections (`bedrock:`, `ollama:`, `pinecone:`, …) override model
ids, URLs, and names — see the
[configuration reference](../reference/configuration.md) for every key.

## The `aws` section is conditional

`aws: {profile, region, account_id}` is required **only while a choice
uses AWS** — the bedrock provider, s3 artifacts, the aws reranker, or an
Amazon OpenSearch domain. Delete it otherwise. If something still needs
it, the config loader's error names the exact choice that does.

## Secrets stay in the environment

`config.yaml` never holds a credential. Keys and connection URLs come from
`.env` (loaded into the process environment) or from real environment
variables — whichever your deployment prefers:

```bash
JINA_API_KEY=…        # reranker: jina
OPENAI_API_KEY=…      # llm/embedding_provider: openai
PINECONE_API_KEY=…    # vector_store: pinecone
QDRANT_URL=…          # vector_store: qdrant (+ QDRANT_API_KEY for cloud)
…
```

Only the selected backend's variables are ever read — a sqlite + ollama
setup needs none at all.

## Precedence everywhere

One rule, applied consistently: **explicit beats preset beats default.**

- A per-call argument (`classify(doc, categories={...})`) beats
  `rules.yaml`'s preset, which beats the open-ended default.
- Passing `{}` explicitly forces the open-ended behavior even when a
  preset exists.
- A config key beats the library default; an absent key means the default.

## Validation

`uv run ingestlib doctor` proves the whole model end to end with real
calls — see the [CLI reference](../reference/cli.md). Config errors
themselves are written to be actionable: a missing `aws` section names the
choices that need it; an unknown `vector_store` lists the valid options.

---

Next: [Providers](providers.md) — what each AI backend serves and what
switching costs.
