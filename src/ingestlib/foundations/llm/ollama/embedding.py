"""Ollama text embeddings (sync and async) via langchain-openai.

The server returns each model's native vector size and ignores a requested
`dimensions` parameter, so the caller's `dimension` must match the
configured model (qwen3-embedding:0.6b → 1024, the pipeline default); a
mismatch raises rather than silently indexing the wrong size. Text only;
`purpose` has no effect (symmetric embeddings).
"""
import asyncio
import threading
import time

from langchain_openai import OpenAIEmbeddings

from ingestlib.config import get_ollama_config
from ingestlib.foundations.llm.ollama.errors import ollama_error_hint
from ingestlib.foundations.llm.types import (
    DEFAULT_DIMENSION,
    EmbeddingDimension,
    EmbeddingPurpose,
)
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_embedder: OpenAIEmbeddings | None = None


def _get_embedder() -> OpenAIEmbeddings:
    global _embedder
    cfg = get_ollama_config()
    with _lock:
        if _embedder is None:
            logger.info(
                "building OpenAIEmbeddings (ollama): base_url=%s model=%s",
                cfg.base_url, cfg.embedding_model_id,
            )
            # check_embedding_ctx_length=False sends the raw string; the
            # default pre-tokenizes with tiktoken and posts token arrays,
            # which the server decodes with a different vocabulary.
            _embedder = OpenAIEmbeddings(
                model=cfg.embedding_model_id,
                api_key="ollama",  # the server needs no key; the client requires one
                base_url=cfg.base_url,
                check_embedding_ctx_length=False,
            )
        return _embedder


def reset_embedders() -> None:
    """Drop the cached instance so the next call rebuilds (e.g. after a config edit)."""
    global _embedder
    with _lock:
        _embedder = None


def embed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Embed text → vector of `dimension` floats (`purpose` has no effect).

    `dimension` must equal the configured model's native size — the server
    cannot resize vectors.
    """
    logger.info("embed_text (ollama): dim=%d input_len=%d", dimension, len(text))
    t0 = time.perf_counter()
    try:
        result = _get_embedder().embed_query(text)
    except Exception as exc:
        hint = ollama_error_hint(exc, get_ollama_config().embedding_model_id)
        if hint:
            raise RuntimeError(f"Ollama embedding failed: {hint}") from exc
        raise
    if len(result) != dimension:
        cfg = get_ollama_config()
        raise ValueError(
            f"{cfg.embedding_model_id} returns {len(result)}-dim vectors, but "
            f"dimension={dimension} was requested — pass the model's native size"
        )
    logger.info("embed_text done: %.2fs returned_dim=%d", time.perf_counter() - t0, len(result))
    return result


async def aembed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Async embed_text() — runs the sync client in a worker thread."""
    return await asyncio.to_thread(embed_text, text, purpose, dimension)
