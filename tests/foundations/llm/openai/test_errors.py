"""Validation guards and error translation.

The bogus-key test provokes a REAL 401 from the live API (free — a
rejected key bills nothing); the other translations are checked against
the SDK's typed exceptions directly, since provoking quota exhaustion or
model 404s would cost real money on a working account."""
import httpx
import openai
import pytest

import ingestlib.config as config_module
from ingestlib.config import OpenAIConfig, get_config
from ingestlib.foundations.llm.openai import reset_embedders, reset_models


def _with_key(monkeypatch, api_key: str):
    current = get_config()
    patched_openai = OpenAIConfig(api_key=api_key, llm_model_id="gpt-5-mini",
                                  embedding_model_id="text-embedding-3-small")
    patched = current.__class__(**{**current.__dict__, "openai": patched_openai})
    monkeypatch.setattr(config_module, "_config", patched)
    reset_models()
    reset_embedders()


def test_chat_without_api_key_raises(monkeypatch):
    _with_key(monkeypatch, "")
    from ingestlib.foundations.llm.openai import chat

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        chat("hi")
    reset_models()  # do not leave key-less instances cached for later tests


def test_embed_without_api_key_raises(monkeypatch):
    _with_key(monkeypatch, "")
    from ingestlib.foundations.llm.openai import embed_text

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        embed_text("hi")
    reset_embedders()


def test_rejected_api_key_names_the_fix(monkeypatch):
    """A bogus key gets a real 401 from OpenAI; the error must say what to do."""
    _with_key(monkeypatch, "sk-proj-definitely-not-a-real-key")
    from ingestlib.foundations.llm.openai import chat

    try:
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            chat("hi")
    finally:
        reset_models()  # never leave bogus-key clients cached


def test_invalid_max_tokens_raises():
    from ingestlib.foundations.llm.openai import chat

    with pytest.raises(ValueError, match="max_tokens"):
        chat("hi", max_tokens=1024)  # type: ignore[arg-type]


def test_invalid_dimension_raises():
    from ingestlib.foundations.llm.openai import embed_text

    with pytest.raises(ValueError, match="dimension"):
        embed_text("hi", dimension=512)  # type: ignore[arg-type]


# ---------- hint mapping against the SDK's typed exceptions ----------


def _response(status: int, code: str) -> httpx.Response:
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status, request=req, json={"error": {"message": code, "code": code}})


def test_hint_rejected_key():
    from ingestlib.foundations.llm.openai.errors import openai_error_hint

    exc = openai.AuthenticationError(
        "bad key", response=_response(401, "invalid_api_key"), body=None,
    )
    assert "OPENAI_API_KEY" in openai_error_hint(exc)


def test_hint_insufficient_quota_says_billing_not_rate_limit():
    from ingestlib.foundations.llm.openai.errors import openai_error_hint

    exc = openai.RateLimitError(
        "insufficient_quota",
        response=_response(429, "insufficient_quota"),
        body={"code": "insufficient_quota"},
    )
    hint = openai_error_hint(exc)
    assert "billing" in hint and "credit" in hint


def test_hint_plain_rate_limit_suggests_waiting():
    from ingestlib.foundations.llm.openai.errors import openai_error_hint

    exc = openai.RateLimitError(
        "rate limited", response=_response(429, "rate_limit_exceeded"), body=None,
    )
    assert "wait" in openai_error_hint(exc)


def test_hint_unknown_model_names_config_keys():
    from ingestlib.foundations.llm.openai.errors import openai_error_hint

    exc = openai.NotFoundError(
        "model not found", response=_response(404, "model_not_found"), body=None,
    )
    hint = openai_error_hint(exc)
    assert "config.yaml" in hint and "llm_model_id" in hint


def test_hint_connection_failure_names_network():
    from ingestlib.foundations.llm.openai.errors import openai_error_hint

    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    exc = openai.APIConnectionError(request=req)
    assert "network" in openai_error_hint(exc)


def test_hint_unrelated_error_returns_none():
    from ingestlib.foundations.llm.openai.errors import openai_error_hint

    assert openai_error_hint(ValueError("nope")) is None
