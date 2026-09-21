"""Fixtures for Ollama backend tests: image bytes + session-cached callers.

Real server, opt-in via RUN_OLLAMA_E2E=1 — needs a local Ollama with the
configured chat and embedding models pulled.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from ingestlib.foundations.llm import Image
from ingestlib.foundations.llm.ollama import chat, chat_structured, embed_text

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_IMAGES_DIR = _TESTS_DIR / "data" / "images"


@pytest.fixture(scope="session")
def photo_bytes() -> bytes:
    return (_IMAGES_DIR / "photo.jpg").read_bytes()


@pytest.fixture(scope="session")
def doc_chart_bytes() -> bytes:
    return (_IMAGES_DIR / "document_chart.png").read_bytes()


class _Ollama:
    """Session-cached callers — identical calls hit the server exactly once."""

    def __init__(self):
        @lru_cache(maxsize=None)
        def _chat(text, system, images_key):
            return chat(text, images=list(images_key) if images_key else None, system=system)

        @lru_cache(maxsize=None)
        def _embed(text, dim):
            return tuple(embed_text(text, dimension=dim))

        self._chat = _chat
        self._embed = _embed

    def chat(self, text: str, images: list[Image] | None = None, system: str | None = None) -> str:
        return self._chat(text, system, tuple(images) if images else None)

    def embed(self, text: str, dim: int = 1024) -> np.ndarray:
        return np.asarray(self._embed(text, dim), dtype=float)

    @staticmethod
    def structured(text, schema):
        return chat_structured(text, schema)


@pytest.fixture(scope="session")
def olm() -> _Ollama:
    return _Ollama()
