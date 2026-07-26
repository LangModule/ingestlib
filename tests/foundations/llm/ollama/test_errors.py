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


def test_hint_persistent_runner_eof_names_the_memory_fix():
    """When the EOF outlasts the retries, the terminal error must point at
    server headroom, not dump a raw 400."""
    import httpx
    import openai

    from ingestlib.foundations.llm.ollama.errors import ollama_error_hint

    req = httpx.Request("POST", "http://localhost:11434/v1/embeddings")
    exc = openai.BadRequestError(
        'do embedding request: Post "http://127.0.0.1:1/v1/embeddings": EOF',
        response=httpx.Response(400, request=req, json={}),
        body={"error": {"message": "EOF"}},
    )
    hint = ollama_error_hint(exc, "qwen3-embedding:0.6b")
    assert "ollama ps" in hint and "restart" in hint


def test_transient_runner_eof_is_retried(monkeypatch, _clean_clients):
    """A local runner can drop a request mid-flight (400 ending in 'EOF') —
    observed in the wild during a corpus backfill. Two real typed failures
    then a success must yield the vector, not an exception.

    Control-flow test: the exception is the SDK's real BadRequestError; the
    embedder is the module seam."""
    import httpx
    import openai

    from ingestlib.foundations.llm.ollama import embedding as embedding_module

    def _eof_error() -> openai.BadRequestError:
        req = httpx.Request("POST", "http://localhost:11434/v1/embeddings")
        resp = httpx.Response(400, request=req, json={"error": {
            "message": 'do embedding request: Post "http://127.0.0.1:1/v1/embeddings": EOF',
        }})
        return openai.BadRequestError(
            'do embedding request: EOF', response=resp,
            body={"error": {"message": "EOF"}},
        )

    calls = {"n": 0}

    class _FlakyEmbedder:
        def embed_query(self, text):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _eof_error()
            return [0.1] * 1024

    monkeypatch.setattr(embedding_module, "_get_embedder", lambda: _FlakyEmbedder())
    monkeypatch.setattr(embedding_module, "_TRANSIENT_BACKOFF_SECONDS", 0.0)

    result = embedding_module.embed_text("retry me")
    assert len(result) == 1024
    assert calls["n"] == 3, "two EOF drops then success — all three attempts used"


def test_invalid_max_tokens_raises():
    from ingestlib.foundations.llm.ollama import chat

    with pytest.raises(ValueError, match="max_tokens"):
        chat("hi", max_tokens=1024)  # type: ignore[arg-type]
