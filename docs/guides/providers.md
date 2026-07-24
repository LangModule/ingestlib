# AI providers

Every LLM and embedding call in the pipeline routes through a provider
switch — two keys in config.yaml, no code changes:

```yaml
llm_provider: bedrock         # bedrock | openai — chat + structured output
embedding_provider: bedrock   # bedrock | openai — text embeddings
```

| | AWS Bedrock (default) | OpenAI |
|---|---|---|
| LLM | Nova 2 Lite | GPT-5 mini |
| Embeddings | Nova 2 multimodal (1024-dim) | text-embedding-3-small |
| Credentials | Your AWS profile | `OPENAI_API_KEY` in `.env` |
| Needs AWS? | Yes (Bedrock model access) | **No** |

The two keys are independent — you can chat on one provider and embed
on the other. The OCR engine is unaffected either way: it is always the
local PaddleOCR-VL server.

## Switching the LLM provider

Safe at any time. Model IDs are overridable per provider
(`bedrock.llm_model_id`, `openai.llm_model_id`) if you want a different
tier.

## Switching the embedding provider — read this first

!!! warning "Changing `embedding_provider` changes the vector space"
    Vectors written by one embedding model are meaningless to queries
    embedded by another. If you switch providers over an existing
    corpus, retrieval quietly degrades to noise.

The safe recipe:

1. Point `vector_store` at a **fresh** store (a different connector, or
   a new database file / collection name).
2. Switch `embedding_provider`.
3. Re-embed the corpus into the fresh store from the artifact store —
   no re-parsing needed (the studio's Backfill button, or the eval
   harness's `--backfill`).

The old store keeps working with the old provider until you delete it.

## Bedrock specifics

- Model access is granted in the **Bedrock console** (Model access
  page), separately from IAM permissions — a common first-run trip.
- Reranking with `reranker: aws` uses `amazon.rerank-v1:0`, which lives
  in `us-west-2` regardless of your main region (handled by the
  `bedrock.rerank_region` default).

## OpenAI specifics

- Structured outputs use the Responses API with strict JSON schemas —
  the same Pydantic-validated results as Bedrock.
- Embeddings are text-only; that is sufficient for the pipeline
  (figures are described in text during parse, and the descriptions are
  what gets embedded).

## Using a provider directly

The pipeline is the main consumer, but the LLM surface is importable on
its own — same functions on both providers, ignoring the config switch:

```python
from ingestlib.foundations.llm import Image
from ingestlib.foundations.llm.openai import chat, chat_structured, embed_text

chat("Read this chart", images=[Image(png_bytes, "png")])   # vision works
embed_text("a chunk of text")
```

(`ingestlib.foundations.llm.bedrock` offers the equivalent Nova surface.)
