# Switch AI providers

Two config keys route every AI call. This page is the practical guide to
changing them safely; the background lives in
[Concepts → Providers](../concepts/providers.md).

## To OpenAI

```yaml
llm_provider: openai
embedding_provider: openai
# openai:                                  # the defaults
#   llm_model_id: gpt-5-mini
#   embedding_model_id: text-embedding-3-small
```

```bash
# .env
OPENAI_API_KEY=sk-…
```

Combined with `artifact_store: local` and `vector_store: sqlite`, an
OpenAI setup needs **no AWS at all** — you can delete the `aws:` section.

## To Ollama (local)

```yaml
llm_provider: ollama
embedding_provider: ollama
```

No key. Full walkthrough: [Run fully local](local-stack.md).

## Mixing providers

The two keys are independent — a common production shape keeps judgment
on a strong cloud model while embeddings run local and free:

```yaml
llm_provider: bedrock
embedding_provider: ollama
```

## The embedding-switch rule

!!! warning "Changing `embedding_provider` changes the vector space"

    Vectors from different embedding models are mutually meaningless — a
    query embedded by the new model finds garbage among vectors from the
    old one. After switching, re-ingest:

    ```python
    for meta in artifacts.list_documents():
        ingest(f"corpus/{meta.filename}", skip_existing=False)
    ```

    Parses are reused from the artifact store, so this repeats **no OCR
    and no parse-stage LLM calls** — just re-embedding and upserting.
    Alternatively, keep one store/namespace per embedding model and switch
    between them.

Switching only `llm_provider` needs none of this — it takes effect on the
next call.

## Verify after any switch

```bash
uv run ingestlib doctor
```

The embedding check prints the new dimension; the LLM check proves a real
round-trip; every failure names its fix.

## Per-call access to a specific backend

The dispatch honors config, but each backend is also importable directly
when you need a one-off call outside the configured route:

```python
from ingestlib.foundations.llm.openai import chat as openai_chat
from ingestlib.foundations.llm.ollama import embed_text as local_embed

openai_chat("Summarize this…")
local_embed("some text")            # 1024-dim, regardless of config
```

---

Next: [Async, notebooks & logging](async-and-logging.md).
