# Installation

## Requirements

- **Python 3.12+**
- **An AI provider** — one of:
    - AWS account with Bedrock model access in `us-east-1` (Nova 2 Lite +
      Nova 2 multimodal embeddings) — the default, or
    - an OpenAI API key (`llm_provider: openai` runs the whole pipeline
      on GPT-5 mini + text-embedding-3; no AWS needed at all).
- **A vector database** — or none: the `sqlite` connector stores vectors
  in a local file with zero setup. The other seven (Pinecone, Qdrant,
  pgvector, MongoDB, Milvus, OpenSearch, Weaviate) each need one
  connection URL or API key. See [Vector stores](../guides/vector-stores.md).
- **A reranker** — a free [Jina](https://jina.ai) key (default), Amazon
  Rerank (`reranker: aws`, same AWS credentials), or `reranker: none`.
- **A GPU for OCR** — Apple Silicon (Metal) or NVIDIA. Parse runs a local
  vision-language model behind an inference server; everything else is
  CPU-fine.

## Install the package

```bash
pip install ingestlib          # or: uv add ingestlib
```

From source:

```bash
git clone https://github.com/LangModule/ingestlib.git
cd ingestlib
uv sync
```

## System dependency: LibreOffice

Only needed for DOCX/PPTX input (they are converted to PDF first):

```bash
brew install --cask libreoffice          # macOS (installs the `soffice` binary)
sudo apt install libreoffice-core libreoffice-writer libreoffice-impress   # Linux
```

PDFs work without it.

## Start the OCR inference server

Parse runs **PaddleOCR-VL-1.6** (a 0.9B vision-language model) behind a
local inference server. The first launch downloads ~1.8 GB of weights;
later launches load from cache in seconds.

=== "Apple Silicon (Metal)"

    ```bash
    uv run python -m mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6
    ```

=== "NVIDIA (vLLM)"

    ```bash
    vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --port 8111
    ```

    Then set `paddle_vl.backend: vllm-server` in config.yaml.

The layout model (PP-DocLayoutV3, ~126 MB) auto-downloads on the first
parse and runs on CPU.

!!! note "Which operations need the server?"
    Only `parse` (and therefore `ingest`). `classify` and `split` in
    standalone mode read the PDF's native text directly, and `retrieve`
    never touches OCR.

## Disk footprint

| Component | Size | Location |
|---|---|---|
| Python dependencies | ~3 GB | your virtualenv |
| PaddleOCR-VL-1.6 weights | ~1.8 GB | `~/.cache/huggingface/hub/` |
| PP-DocLayoutV3 | ~126 MB | `~/.paddlex/official_models/` |
| LibreOffice | ~600 MB | system |

Next: [Configuration](configuration.md).
