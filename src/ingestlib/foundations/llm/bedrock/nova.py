"""Nova 2 Lite chat, thinking-mode, and structured-output primitives (sync and async)."""
import asyncio
import base64
import time
from typing import Any, Literal, NamedTuple, TypeVar

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ingestlib.config import get_bedrock_config
from ingestlib.foundations.llm.bedrock.embedding import ImageFormat
from ingestlib.foundations.llm.bedrock.factory import cache_model, get_model, get_runtime_client
from ingestlib.utils.logger import get_logger


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


logger = get_logger(__name__)

MaxTokens = Literal[8192, 16384, 32768, 65535]
ReasoningEffort = Literal["low", "medium", "high"]

SUPPORTED_MAX_TOKENS: tuple[int, ...] = (8192, 16384, 32768, 65535)
DEFAULT_MAX_TOKENS: MaxTokens = 16384
DEFAULT_THINKING_MAX_TOKENS: MaxTokens = 32768
DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"


class Image(NamedTuple):
    """In-memory image payload: raw bytes + format ("jpeg" | "png" | "webp" | "gif")."""
    data: bytes
    format: ImageFormat


def _validate_max_tokens(max_tokens: int) -> None:
    if max_tokens not in SUPPORTED_MAX_TOKENS:
        raise ValueError(
            f"max_tokens must be one of {SUPPORTED_MAX_TOKENS}, got {max_tokens}"
        )


def get_llm(
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> ChatBedrockConverse:
    """Cached ChatBedrockConverse instance keyed by (max_tokens, temperature).

    LangChain-compatible surface for callers who want chain composition;
    chat()/achat() below talk to Bedrock directly.
    """
    _validate_max_tokens(max_tokens)
    key = f"llm:{max_tokens}:{temperature}"
    cached = get_model(key)
    if cached is not None:
        logger.debug("get_llm cache hit: %s", key)
        return cached  # type: ignore[return-value]

    client = get_runtime_client()
    cfg = get_bedrock_config()
    logger.info("get_llm building new instance: model=%s max_tokens=%d temperature=%s",
                cfg.llm_model_id, max_tokens, temperature)
    instance = ChatBedrockConverse(
        model=cfg.llm_model_id,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cache_model(key, instance, client)
    return instance


def get_llm_with_thinking(
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> ChatBedrockConverse:
    """Cached ChatBedrockConverse with Nova reasoning enabled.

    Keyed by (effort, max_tokens) — except at effort="high", where Nova rejects
    max_tokens entirely, so all max_tokens values share one cache entry.
    """
    _validate_max_tokens(max_tokens)
    # max_tokens is not sent at effort="high" (Nova rejects it) — normalize the
    # key so behaviorally identical instances share one cache entry.
    key_tokens = 0 if effort == "high" else max_tokens
    key = f"llm_thinking:{effort}:{key_tokens}"
    cached = get_model(key)
    if cached is not None:
        logger.debug("get_llm_with_thinking cache hit: %s", key)
        return cached  # type: ignore[return-value]

    client = get_runtime_client()
    cfg = get_bedrock_config()
    logger.info("get_llm_with_thinking building new instance: model=%s effort=%s max_tokens=%d",
                cfg.llm_model_id, effort, max_tokens)
    kwargs: dict[str, Any] = {
        "model": cfg.llm_model_id,
        "client": client,
        "additional_model_request_fields": {
            "reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}
        },
    }
    # Nova rejects both temperature AND maxTokens when maxReasoningEffort == "high"
    if effort != "high":
        kwargs["temperature"] = 0.0
        kwargs["max_tokens"] = max_tokens

    instance = ChatBedrockConverse(**kwargs)
    cache_model(key, instance, client)
    return instance


def _content_blocks(text: str, images: list[Image] | None) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"text": text}]
    if images:
        for img in images:
            blocks.append(
                {"image": {"format": img.format, "source": {"bytes": img.data}}}
            )
    return blocks


def _extract_text(response: dict[str, Any]) -> str:
    content = response["output"]["message"]["content"]
    for block in content:
        if "text" in block:
            return block["text"]
    raise RuntimeError(f"No text block in Nova response: {content!r}")


def _converse(
    *,
    text: str,
    images: list[Image] | None,
    system: str | None,
    inference_config: dict[str, Any],
    additional_fields: dict[str, Any] | None = None,
) -> str:
    client = get_runtime_client()
    cfg = get_bedrock_config()
    kwargs: dict[str, Any] = {
        "modelId": cfg.llm_model_id,
        "messages": [{"role": "user", "content": _content_blocks(text, images)}],
    }
    if inference_config:
        kwargs["inferenceConfig"] = inference_config
    if system:
        kwargs["system"] = [{"text": system}]
    if additional_fields:
        kwargs["additionalModelRequestFields"] = additional_fields

    n_images = len(images) if images else 0
    logger.info(
        "Nova converse start: model=%s prompt_len=%d n_images=%d has_system=%s thinking=%s",
        cfg.llm_model_id, len(text), n_images, system is not None,
        additional_fields is not None,
    )
    t0 = time.perf_counter()
    response = client.converse(**kwargs)
    reply = _extract_text(response)
    logger.info(
        "Nova converse done: %.2fs response_len=%d",
        time.perf_counter() - t0, len(reply),
    )
    return reply


def chat(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Single-turn Nova chat — text plus optional images and system prompt → response text."""
    _validate_max_tokens(max_tokens)
    return _converse(
        text=text,
        images=images,
        system=system,
        inference_config={"maxTokens": max_tokens, "temperature": temperature},
    )


def chat_with_thinking(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> str:
    """Single-turn Nova chat with reasoning enabled — for harder judgment calls."""
    _validate_max_tokens(max_tokens)
    inference: dict[str, Any] = {}
    # Nova rejects both temperature AND maxTokens when maxReasoningEffort == "high"
    if effort != "high":
        inference["maxTokens"] = max_tokens
        inference["temperature"] = 0.0
    return _converse(
        text=text,
        images=images,
        system=system,
        inference_config=inference,
        additional_fields={"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}},
    )


async def achat(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Async chat() — runs the sync Bedrock client in a worker thread."""
    return await asyncio.to_thread(chat, text, images, system, max_tokens, temperature)


async def achat_with_thinking(
    text: str,
    images: list[Image] | None = None,
    system: str | None = None,
    effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
    max_tokens: MaxTokens = DEFAULT_THINKING_MAX_TOKENS,
) -> str:
    """Async chat_with_thinking() — runs the sync Bedrock client in a worker thread."""
    return await asyncio.to_thread(
        chat_with_thinking, text, images, system, effort, max_tokens
    )


def _structured_content_blocks(text: str, images: list[Image] | None) -> list[dict[str, Any]]:
    """LangChain-style content blocks: text plus base64 image blocks."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images or []:
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{img.format}",
                "data": base64.b64encode(img.data).decode(),
            },
        })
    return blocks


def chat_structured(
    text: str,
    schema: type[BaseModelT],
    images: list[Image] | None = None,
    system: str | None = None,
    max_tokens: MaxTokens = DEFAULT_MAX_TOKENS,
) -> BaseModelT:
    """Single-turn Nova chat with schema-enforced output.

    Uses tool-forcing via ChatBedrockConverse.with_structured_output(), so the
    response is a validated instance of `schema` — never free-form text.
    Retries once if the model's output fails schema validation.
    """
    llm = get_llm(max_tokens=max_tokens).with_structured_output(schema)
    messages: list[Any] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=_structured_content_blocks(text, images)))

    logger.info(
        "Nova structured start: schema=%s prompt_len=%d n_images=%d",
        schema.__name__, len(text), len(images) if images else 0,
    )
    t0 = time.perf_counter()
    result = None
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            result = llm.invoke(messages)
        except Exception as exc:  # tool-call parse / schema validation failure
            logger.warning(
                "structured output attempt %d failed (%s: %s)",
                attempt, type(exc).__name__, exc,
            )
            result, last_exc = None, exc
            continue
        if result is not None:
            break
        # with_structured_output returns None when the model skipped the tool
        # call entirely — retry that case too, not just exceptions.
        if attempt == 1:
            logger.warning("structured output attempt 1 returned nothing — retrying once")
    if result is None:
        raise RuntimeError(
            f"Nova produced no valid structured output for schema {schema.__name__} "
            f"after 2 attempts"
        ) from last_exc
    logger.info(
        "Nova structured done: %.2fs schema=%s", time.perf_counter() - t0, schema.__name__,
    )
    return result  # type: ignore[return-value]


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
