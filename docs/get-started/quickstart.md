# Quickstart

From zero to a cited answer. Pick a track: **Local** runs everything on
your machine with no accounts or API keys; **Cloud** uses Amazon Bedrock
for the AI calls.

[Installation](installation.md) should be done first, and the OCR server
running.

## 1. Scaffold a configuration

`ingestlib init` writes the setup files into your current directory —
that's where the library looks for them.

=== "Local — no keys"

    ```bash
    uv run ingestlib init --local
    ```

    This writes a five-choice `config.yaml`: Ollama for LLM + embeddings,
    sqlite for vectors, a local folder for artifacts, no reranker. No
    `.env`, no accounts, nothing leaves your machine.

    Then install [Ollama](https://ollama.com) and pull the two models:

    ```bash
    ollama pull qwen3.5:9b
    ollama pull qwen3-embedding:0.6b
    ```

=== "Cloud — Bedrock"

    ```bash
    uv run ingestlib init
    ```

    This writes `config.yaml` (Bedrock + sqlite + Jina reranking) and a
    `.env` with the key slots. Two things to fill in:

    ```bash
    aws configure --profile your-aws-profile   # Bedrock-enabled credentials
    ```

    Set that profile name in `config.yaml`'s `aws:` section, and put a
    [Jina API key](https://jina.ai/api-dashboard) in `.env` (free tier
    works) — or set `reranker: none` and skip it.

## 2. Verify the stack

```bash
uv run ingestlib doctor
```

Doctor probes every choice with a real call and prints one line per check:

```text
ingestlib doctor
  ✓ python 3.13.13
  ✓ config ./config.yaml (llm=ollama embed=ollama store=sqlite artifacts=local reranker=none)
  ✓ LibreOffice (soffice) found
  ✓ OCR server answering at http://localhost:8111/
  ✓ llm_provider ollama: chat call succeeded
  ✓ embedding_provider ollama: 1024-dim vectors
  - reranker none: retrieve() returns vector order
  ✓ artifacts in local folder ./artifacts
  ✓ vector_store sqlite: reachable

all checks passed — the configured stack is ready
```

Every failed line prints the fix: a wrong AWS profile lists your actual
available profiles, a missing model prints the exact `ollama pull`.

## 3. Ingest a document

```python
from ingestlib.services import ingest

result = ingest("report.pdf")
print(result.status, result.category, result.chunks)
# ingested research_paper 24
```

One call ran the whole pipeline: parse (OCR + chart reading), classify,
split into sections and chunks, embed, and upsert into the vector store —
with every stage's output persisted to the artifact store.

Run it twice and the second call returns `status="skipped"`: documents are
deduplicated by content checksum.

## 4. Ask a question

```python
from ingestlib.services import retrieve

result = retrieve("how were participants recruited?", top_k=5)

for hit in result.hits:
    print(round(hit.vector_score, 3), hit.citation)
    print("   ", hit.chunk.heading)
# 0.94 doc 7b6b95d79149 · p.4 · methods
#      Participant recruitment
```

`result.context` is a numbered, citation-tagged block ready to paste into
an LLM prompt — see [Build cited answers](../how-to/cited-answers.md).

---

That's the whole loop. Next: [What just happened](first-pipeline.md) walks
through everything those two calls produced — on disk and in the store.
