"""Foundation layers the operations are built on.

Sub-packages:
    llm — Bedrock Nova (chat / thinking / structured output), multimodal
          embeddings, and reranking (Jina and Amazon Rerank)
    ocr — PaddleOCR-VL engine (layout + recognition) and the shared region types

Operations (parse / classify / split) compose these; nothing here knows the
operations exist.
"""
