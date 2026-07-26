"""Validation guards and error translation.

Everything here is honestly testable for free: a dead port produces a real
connection refusal (ungated — it tests the ABSENCE of a server), and the
missing-model case gets a real 404 from the live server (gated with the
rest of the e2e suite).
"""
import os

import pytest

import ingestlib.config as config_module
from ingestlib.config import OllamaConfig, get_config
from ingestlib.foundations.llm.ollama import reset_embedders, reset_models


def _with_ollama(monkeypatch, *, base_url=None, llm_model=None, embedding_model=None):
    current = get_config()
    patched_ollama = OllamaConfig(
        base_url=base_url or current.ollama.base_url,
        llm_model_id=llm_model or current.ollama.llm_model_id,
        embedding_model_id=embedding_model or current.ollama.embedding_model_id,
    )
    patched = current.__class__(**{**current.__dict__, "ollama": patched_ollama})
    monkeypatch.setattr(config_module, "_config", patched)
    reset_models()
    reset_embedders()


@pytest.fixture()
def _clean_clients():
    """Never leave clients built against a patched config cached for later tests."""
    yield
    reset_models()
    reset_embedders()


def test_chat_with_dead_server_names_the_fix(monkeypatch, _clean_clients):
    """A real connection refusal must say 'start Ollama', not dump a socket errno."""
    from ingestlib.foundations.llm.ollama import chat

    _with_ollama(monkeypatch, base_url="http://localhost:1/v1")
    with pytest.raises(RuntimeError, match="start Ollama"):
        chat("hi")


def test_embed_with_dead_server_names_the_fix(monkeypatch, _clean_clients):
    from ingestlib.foundations.llm.ollama import embed_text

    _with_ollama(monkeypatch, base_url="http://localhost:1/v1")
    with pytest.raises(RuntimeError, match="start Ollama"):
        embed_text("hi")


@pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_E2E") != "1",
    reason="ollama e2e is opt-in: set RUN_OLLAMA_E2E=1 (needs a local Ollama "
           "with the configured models pulled)",
)
def test_unpulled_model_names_the_pull_command(monkeypatch, _clean_clients):
    """A real 404 from the live server must hand over the pull command."""
    from ingestlib.foundations.llm.ollama import chat

    _with_ollama(monkeypatch, llm_model="definitely-not-pulled:1b")
    with pytest.raises(RuntimeError, match="ollama pull definitely-not-pulled:1b"):
        chat("hi")


def test_hint_unrelated_error_returns_none():
    from ingestlib.foundations.llm.ollama.errors import ollama_error_hint

    assert ollama_error_hint(ValueError("nope"), "m") is None


def test_invalid_max_tokens_raises():
    from ingestlib.foundations.llm.ollama import chat

    with pytest.raises(ValueError, match="max_tokens"):
        chat("hi", max_tokens=1024)  # type: ignore[arg-type]
