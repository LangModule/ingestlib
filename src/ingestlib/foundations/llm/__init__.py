"""Public LLM surface.

- LLM/embedding primitives come from Bedrock (Nova family).
- Rerank is exposed twice with explicit provider suffixes — no ambiguous default:
    aws_rerank / aws_arerank  → amazon.rerank-v1:0 (quota-limited, keep for later)
    jina_rerank / jina_arerank → Jina Reranker API (primary)
"""
from ingestlib.foundations.llm.bedrock import (
    Image,
    achat,
    achat_structured,
    chat_structured,
    achat_with_thinking,
    aembed_image,
    aembed_text,
    chat,
    chat_with_thinking,
    embed_image,
    embed_text,
    get_llm,
    get_llm_with_thinking,
    reset_clients,
)
from ingestlib.foundations.llm.bedrock import arerank as aws_arerank
from ingestlib.foundations.llm.bedrock import rerank as aws_rerank
from ingestlib.foundations.llm.jina import arerank as jina_arerank
from ingestlib.foundations.llm.jina import rerank as jina_rerank

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
    "aws_rerank",
    "aws_arerank",
    "jina_rerank",
    "jina_arerank",
    "reset_clients",
]
