"""Shared LLM vocabulary — the types every backend speaks.

Provider-neutral by design: the chat and embedding surfaces are identical
across bedrock, openai, and ollama, so their parameter types live here and
the backends import them. Nothing in this module touches a client or an
SDK — importing it is free, which keeps a non-AWS pipeline from loading
boto3 just to obtain a type alias.
"""
from typing import Literal, NamedTuple

MaxTokens = Literal[8192, 16384, 32768, 65535]
ReasoningEffort = Literal["low", "medium", "high"]

SUPPORTED_MAX_TOKENS: tuple[int, ...] = (8192, 16384, 32768, 65535)
DEFAULT_MAX_TOKENS: MaxTokens = 16384
DEFAULT_THINKING_MAX_TOKENS: MaxTokens = 32768
DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"

EmbeddingPurpose = Literal["GENERIC_INDEX", "GENERIC_RETRIEVAL", "DOCUMENT_RETRIEVAL"]
EmbeddingDimension = Literal[256, 384, 1024, 3072]
ImageFormat = Literal["jpeg", "png", "webp", "gif"]
ImageDetailLevel = Literal["STANDARD_IMAGE", "DOCUMENT_IMAGE"]

DEFAULT_DIMENSION: EmbeddingDimension = 1024
SUPPORTED_DIMENSIONS: tuple[int, ...] = (256, 384, 1024, 3072)


class Image(NamedTuple):
    """In-memory image payload: raw bytes + format ("jpeg" | "png" | "webp" | "gif")."""
    data: bytes
    format: ImageFormat
