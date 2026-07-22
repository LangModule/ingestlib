"""OpenAI text embeddings (sync and async) via langchain-openai.

text-embedding-3 models take a native `dimensions` parameter, so vectors come
back at the requested size directly — the 1024 default matches what the vector
stores index. Text only; `purpose` has no effect (symmetric embeddings).
"""
import asyncio
import threading
import time

from langchain_openai import OpenAIEmbeddings

from ingestlib.config import get_openai_config
from ingestlib.foundations.llm.bedrock.embedding import (
    DEFAULT_DIMENSION,
    EmbeddingDimension,
    EmbeddingPurpose,
    SUPPORTED_DIMENSIONS,
)
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_embedder_cache: dict[int, OpenAIEmbeddings] = {}


def _validate_dimension(dimension: int) -> None:
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(
            f"dimension must be one of {SUPPORTED_DIMENSIONS}, got {dimension}"
        )


def _embedder(dimension: int) -> OpenAIEmbeddings:
    cfg = get_openai_config()
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to .env")
    with _lock:
        instance = _embedder_cache.get(dimension)
        if instance is None:
            logger.info(
                "building OpenAIEmbeddings: model=%s dim=%d",
                cfg.embedding_model_id, dimension,
            )
            instance = OpenAIEmbeddings(
                model=cfg.embedding_model_id,
                api_key=cfg.api_key,
                dimensions=dimension,
            )
            _embedder_cache[dimension] = instance
        return instance


def reset_embedders() -> None:
    """Drop cached instances so the next call rebuilds (e.g. after key rotation)."""
    with _lock:
        _embedder_cache.clear()


def embed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Embed text → vector of `dimension` floats (`purpose` has no effect)."""
    _validate_dimension(dimension)
    logger.info("embed_text (openai): dim=%d input_len=%d", dimension, len(text))
    t0 = time.perf_counter()
    result = _embedder(dimension).embed_query(text)
    logger.info("embed_text done: %.2fs returned_dim=%d", time.perf_counter() - t0, len(result))
    return result


async def aembed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Async embed_text() — runs the sync client in a worker thread."""
    return await asyncio.to_thread(embed_text, text, purpose, dimension)
