"""Fixtures for Nova 2 Lite chat tests: image bytes + session-cached chat helper."""
from functools import lru_cache
from pathlib import Path

import pytest

from ingestlib.foundations.llm import Image, chat, chat_with_thinking

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_IMAGES_DIR = _TESTS_DIR / "data" / "images"


@pytest.fixture(scope="session")
def photo_bytes() -> bytes:
    return (_IMAGES_DIR / "photo.jpg").read_bytes()


@pytest.fixture(scope="session")
def doc_text_bytes() -> bytes:
    return (_IMAGES_DIR / "document_text.png").read_bytes()


@pytest.fixture(scope="session")
def doc_chart_bytes() -> bytes:
    return (_IMAGES_DIR / "document_chart.png").read_bytes()


class _LLM:
    """Session-cached chat helper — identical calls hit Bedrock exactly once."""

    def __init__(self):
        @lru_cache(maxsize=None)
        def _chat(text, system, max_tokens, temperature, images_key):
            return chat(
                text,
                images=list(images_key) if images_key else None,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        @lru_cache(maxsize=None)
        def _chat_thinking(text, system, effort, max_tokens, images_key):
            return chat_with_thinking(
                text,
                images=list(images_key) if images_key else None,
                system=system,
                effort=effort,
                max_tokens=max_tokens,
            )

        self._chat = _chat
        self._chat_thinking = _chat_thinking

    def chat(
        self,
        text: str,
        images: list[Image] | None = None,
        system: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> str:
        images_key = tuple(images) if images else None
        return self._chat(text, system, max_tokens, temperature, images_key)

    def chat_with_thinking(
        self,
        text: str,
        images: list[Image] | None = None,
        system: str | None = None,
        effort: str = "medium",
        max_tokens: int = 32768,
    ) -> str:
        images_key = tuple(images) if images else None
        return self._chat_thinking(text, system, effort, max_tokens, images_key)


@pytest.fixture(scope="session")
def llm() -> _LLM:
    return _LLM()
