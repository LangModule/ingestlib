"""Bedrock backend: Nova LLM/embedding + amazon.rerank-v1:0. Clients built in factory.py."""
from ingestlib.foundations.llm.bedrock.embedding import (
    aembed_image,
    aembed_text,
    embed_image,
    embed_text,
)
from ingestlib.foundations.llm.bedrock.factory import reset_clients
from ingestlib.foundations.llm.bedrock.nova import (
    Image,
    achat,
    achat_structured,
    achat_with_thinking,
    chat,
    chat_structured,
    chat_with_thinking,
    get_llm,
    get_llm_with_thinking,
)
from ingestlib.foundations.llm.bedrock.rerank import arerank, rerank

__all__ = [
    "Image",
    "get_llm",
    "get_llm_with_thinking",
    "chat",
    "chat_with_thinking",
    "achat",
    "chat_structured",
    "achat_structured",
    "achat_with_thinking",
    "embed_text",
    "embed_image",
    "aembed_text",
    "aembed_image",
    "rerank",
    "arerank",
    "reset_clients",
]
