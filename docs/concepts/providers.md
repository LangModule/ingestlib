# Providers

Every LLM and embedding call in the pipeline is served by one of three
interchangeable backends. Two config keys select who does what; the code
calling them never knows which backend answered.

```yaml
llm_provider: bedrock        # chart reading, review, classify, chunk boundaries
embedding_provider: bedrock  # chunk + query embeddings
```

They're independent — chat on one provider, embeddings on another is a
supported (and sometimes ideal) setup.

## The three backends

| | Bedrock (default) | OpenAI | Ollama |
|---|---|---|---|
| Chat + vision | Nova 2 Lite | GPT-5 (`gpt-5-mini`) | Qwen3.5 9B (any local model) |
| Structured output | ✓ schema-enforced | ✓ schema-enforced | ✓ schema-enforced (GGUF builds) |
| Text embeddings | Nova multimodal, 1024-dim | text-embedding-3, 1024-dim | qwen3-embedding, 1024-dim |
| Image embeddings | ✓ | — | — |
| Needs | AWS account, Bedrock model access | `OPENAI_API_KEY` | a local [Ollama](https://ollama.com) server — no key |
| Data leaves your infra | AWS | OpenAI | **never** |

Model ids are overridable per backend (`bedrock:`, `openai:`, `ollama:`
sections). The ollama backend speaks the OpenAI-compatible API, so any
server that implements it — vLLM, LM Studio — works by pointing
`ollama.base_url` at it.

## The vision requirement

Parse's enrichment (charts → data tables, figure descriptions) and
image-file classification send **images** to the LLM. All three reference
models are vision-capable; if you swap in your own model, it must be too.

!!! warning "Ollama: use GGUF builds, not `-mlx`"

    Ollama's MLX engine silently delivers **no images** to the model and
    ignores schema enforcement — charts get hallucinated instead of read.
    The standard GGUF builds (`qwen3.5:9b`) handle both correctly. This is
    verified behavior, not superstition.

## Switching the LLM provider

Free. The next call routes to the new backend — no migration, nothing to
rebuild. Quality differs (a local 9B won't match cloud models on dense
chart pages), but nothing breaks.

## Switching the embedding provider

**Not free.** Embeddings from different models live in different vector
spaces — a query embedded by model B finds nothing meaningful among
vectors from model A. After changing `embedding_provider`:

1. Re-ingest your corpus (`skip_existing=False`), or
2. Keep separate stores/namespaces per embedding model

The stored ingest manifest records which dimension and store each document
was ingested with, so a mismatch is diagnosable after the fact.

## Reranking is chosen separately

`reranker: jina | aws | none` picks retrieve()'s second stage — a
cross-encoder that reads full text and re-orders candidates:

- **jina** (default) — `jina-reranker-v3` via API, free tier available
- **aws** — Amazon Rerank via Bedrock, same AWS credentials
- **none** — vector order as-is; also the fully-local choice

A reranker failure never kills a query: retrieve logs a warning and
returns vector order.

---

Next: [Storage model](storage.md) — artifacts, vectors, and hybrid search.
