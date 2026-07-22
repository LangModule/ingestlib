"""OpenAI backend: GPT-5 LLM + text-embedding-3 embeddings.

API key from OPENAI_API_KEY in .env; model IDs from config.yaml's `openai`
section. Not provided (no OpenAI equivalent exists): image embeddings and
a reranker.
"""
from ingestlib.foundations.llm.openai.embedding import (
    aembed_text,
    embed_text,
    reset_embedders,
)
from ingestlib.foundations.llm.openai.mini import (
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
