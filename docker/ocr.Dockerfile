# syntax=docker/dockerfile:1
# OCR model server — vLLM serving PaddleOCR-VL-1.6, reached by the app at
# http://ocr:8111. NVIDIA GPU only. Weights download to the mounted HF cache on
# first run (the ocr service in infra/docker-compose.stack.yml).
FROM vllm/vllm-openai:latest

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8111
# PaddleOCR-VL ships custom modeling code → --trust-remote-code.
ENTRYPOINT ["vllm", "serve", "PaddlePaddle/PaddleOCR-VL-1.6", \
            "--trust-remote-code", \
            "--host", "0.0.0.0", "--port", "8111"]
