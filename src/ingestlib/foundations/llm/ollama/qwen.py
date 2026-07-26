"""Qwen3.5 chat, thinking-mode, and structured-output primitives (sync and async).

Built on langchain-openai's ChatOpenAI against an OpenAI-compatible server
(Ollama by default; any compatible server via `ollama.base_url`). Runs on
the chat-completions API — these servers do not implement the Responses
API. Plain chat disables thinking entirely: locally, a thinking chain can
take minutes where the plain answer takes a second, and the pipeline's
judgment calls do not need it. The thinking variants raise it.
"""
import asyncio
import base64
import threading
import time
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ingestlib.config import get_ollama_config
from ingestlib.foundations.llm.ollama.errors import ollama_error_hint
from ingestlib.foundations.llm.types import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_THINKING_MAX_TOKENS,
    SUPPORTED_MAX_TOKENS,
    Image,
    MaxTokens,
    ReasoningEffort,
)
from ingestlib.utils.logger import get_logger


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)

logger = get_logger(__name__)

_lock = threading.Lock()
_model_cache: dict[str, ChatOpenAI] = {}

# "none" switches the model's thinking off server-side; without it every
# call pays a long reasoning chain before the first answer token.
_CHAT_EFFORT = "none"


def _validate_max_tokens(max_tokens: int) -> None:
    if max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(
            f"max_tokens must be one of {SUPPORTED_MAX_TOKENS}, got {max_tokens}"
        )


def _build(effort: str, max_tokens: int) -> ChatOpenAI:
    cfg = get_ollama_config()
    logger.info(
        "building ChatOpenAI (ollama): base_url=%s model=%s effort=%s max_tokens=%d",
        cfg.base_url, cfg.llm_model_id, effort, max_tokens,
    )
    # The server needs no key, but the client requires one — any value works.
    # temperature pins to 0: local models honor it (unlike GPT-5), and the
    # pipeline's judgment calls need deterministic answers.
    return ChatOpenAI(
        model=cfg.llm_model_id,
        api_key="ollama",
        base_url=cfg.base_url,
        use_responses_api=False,
        max_completion_tokens=max_tokens,
        reasoning_effort=effort,
        temperature=0.0,
    )


def _cached(effort: str, max_tokens: int) -> ChatOpenAI:
    key = f"{effort}:{max_tokens}"
    with _lock:
        instance = _model_cache.get(key)
        if instance is None:
            instance = _build(effort, max_tokens)
            _model_cache[key] = instance
        return instance


def get_llm(
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Cached ChatOpenAI instance with thinking disabled.

    LangChain-compatible surface for callers who want chain composition.
    `temperature` has no effect — the client pins 0.
    """
    _validate_max_tokens(max_tokens)
    return _cached(_CHAT_EFFORT, max_tokens)


def get_llm_with_thinking(
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> ChatOpenAI:
    """Cached ChatOpenAI with thinking enabled, keyed by (effort, max_tokens)."""
    _validate_max_tokens(max_tokens)
    return _cached(effort, max_tokens)


def reset_models() -> None:
    """Drop cached instances so the next call rebuilds (e.g. after a config edit)."""
    with _lock:
        _model_cache.clear()


def _content_blocks(text: str, images: list[Image] | None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images or []:
        blocks.append({
            "type": "image",
            "base64": base64.b64encode(img.data).decode(),
            "mime_type": f"image/{img.format}",
        })
    return blocks


def _messages(
    text: str, images: list[Image] | None, system: str | None
) -> list[Any]:
    messages: list[Any] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=_content_blocks(text, images)))
    return messages


def _extract_text(content: Any) -> str:
    """Response content → plain text (tolerates block lists and plain strings)."""
    if isinstance(content, str):
        return content
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _invoke(llm: ChatOpenAI, text: str, images: list[Image] | None, system: str | None) -> str:
    cfg = get_ollama_config()
    n_images = len(images) if images else 0
    logger.info(
        "Ollama chat start: model=%s prompt_len=%d n_images=%d has_system=%s",
        cfg.llm_model_id, len(text), n_images, system is not None,
    )
    t0 = time.perf_counter()
    try:
        response = llm.invoke(_messages(text, images, system))
    except Exception as exc:
        hint = ollama_error_hint(exc, cfg.llm_model_id)
        if hint:
            raise RuntimeError(f"Ollama chat failed: {hint}") from exc
        raise
    reply = _extract_text(response.content)
    logger.info(
        "Ollama chat done: %.2fs response_len=%d", time.perf_counter() - t0, len(reply),
    )
    return reply


def chat(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Single-turn Qwen chat — text plus optional images and system prompt → response text."""
    _validate_max_tokens(max_tokens)
    return _invoke(_cached(_CHAT_EFFORT, max_tokens), text, images, system)


def chat_with_thinking(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> str:
    """Single-turn Qwen chat with thinking enabled — for harder judgment calls."""
    _validate_max_tokens(max_tokens)
    return _invoke(_cached(effort, max_tokens), text, images, system)


def chat_structured(
    text: str,
    schema: type[BaseModelT],
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
) -> BaseModelT:
    """Single-turn Qwen chat with schema-enforced output.

    Uses ChatOpenAI.with_structured_output(method="json_schema") — the server
    constrains generation to the schema, so the response is a validated
    instance of `schema`, never free-form text. Retries once on failure.
    """
    llm = _cached(_CHAT_EFFORT, max_tokens).with_structured_output(
        schema, method="json_schema"
    )
    messages = _messages(text, images, system)

    logger.info(
        "Ollama structured start: schema=%s prompt_len=%d n_images=%d",
        schema.__name__, len(text), len(images) if images else 0,
    )
    t0 = time.perf_counter()
    result = None
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            result = llm.invoke(messages)
        except Exception as exc:  # schema validation / parse failure
            hint = ollama_error_hint(exc, get_ollama_config().llm_model_id)
            if hint:  # a dead server / missing model doesn't improve on retry
                raise RuntimeError(f"Ollama chat failed: {hint}") from exc
            logger.warning(
                "structured output attempt %d failed (%s: %s)",
                attempt, type(exc).__name__, exc,
            )
            result, last_exc = None, exc
            continue
        if result is not None:
            break
        if attempt == 1:
            logger.warning("structured output attempt 1 returned nothing — retrying once")
    if result is None:
        raise RuntimeError(
            f"Ollama produced no valid structured output for schema {schema.__name__} "
            f"after 2 attempts"
        ) from last_exc
    logger.info(
        "Ollama structured done: %.2fs schema=%s", time.perf_counter() - t0, schema.__name__,
    )
    return result  # type: ignore[return-value]


async def achat(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Async chat() — runs the sync client in a worker thread."""
    return await asyncio.to_thread(chat, text, images, system, max_tokens, temperature)


async def achat_with_thinking(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> str:
    """Async chat_with_thinking() — runs the sync client in a worker thread."""
    return await asyncio.to_thread(
        chat_with_thinking, text, images, system, effort, max_tokens
    )


async def achat_structured(
    text: str,
    schema: type[BaseModelT],
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
) -> BaseModelT:
    """Async chat_structured() — runs the sync client in a worker thread."""
    return await asyncio.to_thread(
        chat_structured, text, schema, images, system, max_tokens
    )
