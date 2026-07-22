"""Foundation layers the operations are built on.

Sub-packages:
    llm — chat / thinking / structured output and text embeddings, served
          by Bedrock Nova or OpenAI GPT-5 (config.yaml's llm_provider /
          embedding_provider), plus reranking (Jina and Amazon Rerank)
    ocr — PaddleOCR-VL engine (layout + recognition) and the shared region types

Operations (parse / classify / split) compose these; nothing here knows the
operations exist.
"""
