"""Client-side guards — no API calls, always run."""
import pytest

import ingestlib.config as config_module
from ingestlib.config import OpenAIConfig, get_config
from ingestlib.foundations.llm.openai import reset_embedders, reset_models


def _with_empty_key(monkeypatch):
    current = get_config()
    empty = OpenAIConfig(api_key="", llm_model_id="gpt-5-mini",
                         embedding_model_id="text-embedding-3-small")
    patched = current.__class__(**{**current.__dict__, "openai": empty})
    monkeypatch.setattr(config_module, "_config", patched)
    reset_models()
    reset_embedders()


def test_chat_without_api_key_raises(monkeypatch):
    _with_empty_key(monkeypatch)
    from ingestlib.foundations.llm.openai import chat

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        chat("hi")
    reset_models()  # do not leave key-less instances cached for later tests


def test_embed_without_api_key_raises(monkeypatch):
    _with_empty_key(monkeypatch)
    from ingestlib.foundations.llm.openai import embed_text

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        embed_text("hi")
    reset_embedders()


def test_invalid_max_tokens_raises():
    from ingestlib.foundations.llm.openai import chat

    with pytest.raises(ValueError, match="max_tokens"):
        chat("hi", max_tokens=1024)  # type: ignore[arg-type]


def test_invalid_dimension_raises():
    from ingestlib.foundations.llm.openai import embed_text

    with pytest.raises(ValueError, match="dimension"):
        embed_text("hi", dimension=512)  # type: ignore[arg-type]
