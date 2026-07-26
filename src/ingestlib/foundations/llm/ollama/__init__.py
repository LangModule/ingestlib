"""Ollama backend: local models through an OpenAI-compatible server.

No API key — the server runs on your machine (or network); `ollama.base_url`
in config.yaml points at it, and any OpenAI-compatible server (vLLM,
LM Studio) works the same way. Model IDs come from the same section;
reference stack: qwen3.5:9b + qwen3-embedding:0.6b. Not provided:
image embeddings and a reranker.
"""
from ingestlib.foundations.llm.ollama.embedding import (
    aembed_text,
    embed_text,
    reset_embedders,
)
from ingestlib.foundations.llm.ollama.qwen import (
    achat,
    achat_structured,
    achat_with_thinking,
    chat,
    chat_structured,
    chat_with_thinking,
    get_llm,
    get_llm_with_thinking,
    reset_models,
)

__all__ = [
    "get_llm",
    "get_llm_with_thinking",
    "chat",
    "chat_with_thinking",
    "achat",
    "chat_structured",
    "achat_structured",
    "achat_with_thinking",
    "embed_text",
    "aembed_text",
    "reset_models",
    "reset_embedders",
]
