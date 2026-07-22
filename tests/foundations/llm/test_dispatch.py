"""Provider dispatch in foundations.llm — pure, always run.

The public surface must route each call to the backend named in config
(llm_provider for chat, embedding_provider for embeddings), at call time,
with arguments passed through untouched. Backends are monkeypatched, so
nothing here builds a client or reaches the network.
"""
from dataclasses import replace

import pytest

import ingestlib.foundations.llm as llm
from ingestlib.config import get_config


@pytest.fixture()
def set_providers(monkeypatch):
    """Return a function that pins llm_provider/embedding_provider for the test."""
    def _set(llm_provider: str = "bedrock", embedding_provider: str = "bedrock"):
        pinned = replace(
            get_config(),
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )
        monkeypatch.setattr("ingestlib.foundations.llm.get_config", lambda: pinned)
    return _set


def test_backend_selection(set_providers):
    from ingestlib.foundations.llm import bedrock, openai

    set_providers()
    assert llm._llm() is bedrock and llm._embedder() is bedrock

    set_providers(llm_provider="openai", embedding_provider="openai")
    assert llm._llm() is openai and llm._embedder() is openai


def test_llm_and_embedding_providers_are_independent(set_providers):
    from ingestlib.foundations.llm import bedrock, openai

    set_providers(llm_provider="openai", embedding_provider="bedrock")
    assert llm._llm() is openai and llm._embedder() is bedrock


def test_unknown_provider_raises(set_providers):
    set_providers(llm_provider="cohere")
    with pytest.raises(ValueError, match="cohere"):
        llm._llm()


def test_chat_dispatches_with_arguments_intact(set_providers, monkeypatch):
    set_providers(llm_provider="openai")
    calls = []
    monkeypatch.setattr(
        "ingestlib.foundations.llm.openai.chat",
        lambda *args: calls.append(args) or "routed",
    )
    reply = llm.chat("hello", None, "sys", 8192)
    assert reply == "routed"
    assert calls == [("hello", None, "sys", 8192, 0.0)]


def test_embed_text_dispatches_with_arguments_intact(set_providers, monkeypatch):
    set_providers(embedding_provider="openai")
    calls = []
    monkeypatch.setattr(
        "ingestlib.foundations.llm.openai.embed_text",
        lambda *args: calls.append(args) or [0.0],
    )
    vector = llm.embed_text("some text", "GENERIC_RETRIEVAL", 256)
    assert vector == [0.0]
    assert calls == [("some text", "GENERIC_RETRIEVAL", 256)]


async def test_achat_structured_dispatches(set_providers, monkeypatch):
    from pydantic import BaseModel

    class Shape(BaseModel):
        answer: str

    set_providers(llm_provider="openai")

    async def fake(*args):
        return Shape(answer="ok")

    monkeypatch.setattr("ingestlib.foundations.llm.openai.achat_structured", fake)
    result = await llm.achat_structured("q", Shape)
    assert result.answer == "ok"


def test_embed_image_on_openai_raises_not_implemented(set_providers):
    set_providers(embedding_provider="openai")
    with pytest.raises(NotImplementedError, match="image"):
        llm.embed_image(b"png-bytes", "png")


def test_embed_image_on_bedrock_still_dispatches(set_providers, monkeypatch):
    set_providers(embedding_provider="bedrock")
    monkeypatch.setattr(
        "ingestlib.foundations.llm.bedrock.embed_image",
        lambda *args: [1.0],
    )
    assert llm.embed_image(b"png-bytes", "png") == [1.0]
