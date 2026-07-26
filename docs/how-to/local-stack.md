# Run fully local

The complete pipeline — OCR, LLM judgment, embeddings, vectors, artifacts —
with **nothing leaving your machine**. No API keys, no cloud accounts. This
is the configuration the self-hosted pitch is actually about.

## The stack

| Piece | Runs on |
|---|---|
| OCR | PaddleOCR-VL on your GPU (already local in every setup) |
| LLM + embeddings | [Ollama](https://ollama.com) serving Qwen models |
| Vectors | sqlite — one local file |
| Artifacts | a local folder |
| Reranker | none |

## Set it up

```bash
uv run ingestlib init --local
```

writes the entire config:

```yaml
llm_provider: ollama
embedding_provider: ollama
vector_store: sqlite
artifact_store: local
reranker: none
```

Install Ollama and pull the reference models:

```bash
ollama pull qwen3.5:9b            # vision LLM — judgment tasks
ollama pull qwen3-embedding:0.6b  # 1024-dim embeddings
```

Start the [OCR server](../get-started/installation.md#3-the-ocr-inference-server),
then prove the whole stack:

```bash
uv run ingestlib doctor
```

Every check green means real calls succeeded — an actual chat round-trip,
an actual embedding. From here `ingest()` and `retrieve()` work exactly as
in the [Quickstart](../get-started/quickstart.md); no code changes, ever.

## The rules that keep it working

!!! warning "GGUF builds only — never `-mlx`"

    Ollama's MLX engine silently delivers **no images** to the model and
    ignores schema enforcement: charts get *hallucinated* instead of read,
    with no error. The standard GGUF builds handle vision and structured
    output correctly. `qwen3.5:9b` is GGUF; anything tagged `-mlx` is not
    usable for this pipeline.

- **Pick a vision-capable model** if you swap the LLM — parse sends chart
  and figure images to it.
- **Thinking stays off** — the backend pins `reasoning_effort` off for
  pipeline calls (a thinking-enabled local call can take minutes instead
  of a second).
- **Any OpenAI-compatible server works** — vLLM, LM Studio: point
  `ollama.base_url` at it and set your model ids:

```yaml
ollama:
  base_url: http://localhost:8000/v1
  llm_model_id: your-model
  embedding_model_id: your-embedding-model
```

## Honest expectations

Split the question in two:

**Embeddings — measured equal to cloud.** On the retrieval eval (22
ground-truth questions), the local `qwen3-embedding:0.6b` matched or beat
the cloud embeddings: a perfect 1.00 on reranked hit@1/3/5 and MRR.
Retrieval quality is not the compromise here.

**LLM judgment — good, with known limits.** A local 9B is genuinely
capable at classification, section labeling, and chunk boundaries, and
reads ordinary charts correctly. It will not match Nova or GPT-5 on
*dense* chart pages and complex figures. Run your own documents through
both stacks before committing; quality is corpus-dependent.

The one remaining external call in a default `--local` setup: none. If you
later enable `reranker: jina`, that's the single piece that leaves the
machine — `reranker: none` (the preset's choice) closes it.

---

Next: [Connect a vector store](vector-stores.md) when one sqlite file
isn't enough.
