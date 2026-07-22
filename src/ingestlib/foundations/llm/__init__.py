"""Public LLM surface — dispatched to the configured backend.

config.yaml picks who serves each call family; operations and services
import from here and never know which backend answered:

    llm_provider: bedrock | openai        → chat / thinking / structured / get_llm
    embedding_provider: bedrock | openai  → embed_text (and embed_image on bedrock)

Both backends expose identical signatures, so dispatch is a per-call config
read — no client is built until a call actually happens. Switching
embedding_provider changes the vector space: re-ingest (or --backfill) after.
Image embeddings exist only on bedrock. Rerank keeps its explicit provider
suffixes (aws_rerank / jina_rerank); retrieve() picks via config.yaml's
`reranker` key.
"""
from types import ModuleType
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from ingestlib.config import get_config
from ingestlib.foundations.llm.bedrock import arerank as aws_arerank
from ingestlib.foundations.llm.bedrock import rerank as aws_rerank
from ingestlib.foundations.llm.bedrock.embedding import (
    DEFAULT_DIMENSION,
    EmbeddingDimension,
    EmbeddingPurpose,
    ImageDetailLevel,
    ImageFormat,
)
from ingestlib.foundations.llm.bedrock.factory import reset_clients
from ingestlib.foundations.llm.bedrock.nova import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING_MAX_TOKENS,
    Image,
    MaxTokens,
    ReasoningEffort,
)
from ingestlib.foundations.llm.jina import arerank as jina_arerank
from ingestlib.foundations.llm.jina import rerank as jina_rerank

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def _backend(provider: str) -> ModuleType:
    if provider == "bedrock":
        from ingestlib.foundations.llm import bedrock

        return bedrock
    if provider == "openai":
        from ingestlib.foundations.llm import openai

        return openai
    raise ValueError(f"unknown LLM provider {provider!r} — expected 'bedrock' or 'openai'")


def _llm() -> ModuleType:
    return _backend(get_config().llm_provider)


def _embedder() -> ModuleType:
    return _backend(get_config().embedding_provider)


def get_llm(
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Cached chat model from the configured llm_provider, for chain composition."""
    return _llm().get_llm(max_tokens, temperature)


def get_llm_with_thinking(
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> BaseChatModel:
    """Cached chat model with reasoning enabled, from the configured llm_provider."""
    return _llm().get_llm_with_thinking(effort, max_tokens)


def chat(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Single-turn chat — text plus optional images and system prompt → response text."""
    return _llm().chat(text, images, system, max_tokens, temperature)


def chat_with_thinking(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> str:
    """Single-turn chat with reasoning raised — for harder judgment calls."""
    return _llm().chat_with_thinking(text, images, system, effort, max_tokens)


def chat_structured(
    text: str,
    schema: type[BaseModelT],
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
) -> BaseModelT:
    """Single-turn chat with schema-enforced output — returns a validated instance."""
    return _llm().chat_structured(text, schema, images, system, max_tokens)


async def achat(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Async chat()."""
    return await _llm().achat(text, images, system, max_tokens, temperature)


async def achat_with_thinking(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> str:
    """Async chat_with_thinking()."""
    return await _llm().achat_with_thinking(text, images, system, effort, max_tokens)


async def achat_structured(
    text: str,
    schema: type[BaseModelT],
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
) -> BaseModelT:
    """Async chat_structured()."""
    return await _llm().achat_structured(text, schema, images, system, max_tokens)


def embed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Embed text via the configured embedding_provider → vector of `dimension` floats."""
    return _embedder().embed_text(text, purpose, dimension)


async def aembed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Async embed_text()."""
    return await _embedder().aembed_text(text, purpose, dimension)


def _image_embedder() -> ModuleType:
    backend = _embedder()
    if not hasattr(backend, "embed_image"):
        raise NotImplementedError(
            f"embedding_provider '{get_config().embedding_provider}' has no image "
            "embeddings — only bedrock does"
        )
    return backend


def embed_image(
    data: bytes,
    format: ImageFormat,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
    detail_level: ImageDetailLevel = "STANDARD_IMAGE",
) -> list[float]:
    """Embed an image → vector of `dimension` floats (bedrock only)."""
    return _image_embedder().embed_image(data, format, purpose, dimension, detail_level)


async def aembed_image(
    data: bytes,
    format: ImageFormat,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
    detail_level: ImageDetailLevel = "STANDARD_IMAGE",
) -> list[float]:
    """Async embed_image() (bedrock only)."""
    return await _image_embedder().aembed_image(
        data, format, purpose, dimension, detail_level
    )


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
