# Installation

ingestlib needs Python 3.12+, one system dependency for Office formats, and
an OCR inference server for parsing. This page gets all three in place.

## 1. Install the package

=== "uv"

    ```bash
    uv add ingestlib
    ```

=== "pip"

    ```bash
    pip install ingestlib
    ```

=== "From source"

    ```bash
    git clone https://github.com/LangModule/ingestlib.git
    cd ingestlib
    uv sync
    ```

This installs the `ingestlib` command alongside the library — you'll use it
in the [Quickstart](quickstart.md) to scaffold and verify your setup.

## 2. LibreOffice (DOCX/PPTX only)

Office documents are converted to PDF before parsing, which needs
LibreOffice on the machine. PDFs and images work without it.

=== "macOS"

    ```bash
    brew install --cask libreoffice
    ```

=== "Linux"

    ```bash
    sudo apt install libreoffice-core libreoffice-writer libreoffice-impress
    ```

## 3. The OCR inference server

`parse()` (and therefore `ingest()`) runs
[PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) —
a 0.9B vision model — behind an inference server on your own GPU. The first
launch downloads ~1.8 GB of weights; later launches load from cache in
seconds.

=== "Apple Silicon"

    Served by mlx-vlm on the Metal GPU (installed automatically with
    ingestlib on macOS/arm64):

    ```bash
    uv run python -m mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6
    ```

=== "NVIDIA"

    Served by [vLLM](https://docs.vllm.ai):

    ```bash
    vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --port 8111
    ```

    Then set `paddle_vl.backend: vllm-server` in your config.yaml.

!!! note "Only parsing needs the server"

    `classify()`, `split()`, and `retrieve()` all run without the OCR
    server. If you only see a server warning from `ingestlib doctor`, the
    rest of the pipeline still works.

A second, smaller layout model (PP-DocLayoutV3, ~126 MB) downloads
automatically on your first parse.

## Disk footprint

| Component | Size | Location |
|---|---|---|
| Python dependencies | ~1.6 GB | your environment |
| PaddleOCR-VL-1.6 weights | ~1.8 GB | `~/.cache/huggingface/hub/` |
| PP-DocLayoutV3 | ~126 MB | `~/.paddlex/official_models/` |
| LibreOffice | ~600 MB | system |

---

Next: [Quickstart](quickstart.md) — configure a stack and get your first
cited answer.
