"""Validation guards and error translation.

The translation tests provoke REAL failures (a bogus key against the live
API, an unreachable server) rather than mocking responses; the 402/429
branches stay untested — triggering them would burn the real quota."""
import importlib

import pytest

from ingestlib.config import JinaConfig
from ingestlib.foundations.llm import jina_rerank


def test_empty_docs_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        jina_rerank("query", [])


def test_missing_api_key_raises_runtime_error(monkeypatch):
    """rerank() must fail loudly if JINA_API_KEY is unset, before any HTTP call."""
    # jina/__init__.py re-exports `rerank` as a function, shadowing the submodule name;
    # grab the module object explicitly so monkeypatch targets the real namespace.
    rerank_module = importlib.import_module("ingestlib.foundations.llm.jina.rerank")
    empty = JinaConfig(
        api_key="",
        base_url="https://api.jina.ai/v1",
        rerank_model_id="jina-reranker-v3",
    )
    monkeypatch.setattr(rerank_module, "get_jina_config", lambda: empty)
    with pytest.raises(RuntimeError, match="JINA_API_KEY is not set"):
        jina_rerank("query", ["doc"])


def test_rejected_api_key_names_the_fix(monkeypatch):
    """A bogus key gets a real 401 from Jina; the error must say what to do."""
    rerank_module = importlib.import_module("ingestlib.foundations.llm.jina.rerank")
    bogus = JinaConfig(
        api_key="jina_definitely-not-a-real-key",
        base_url="https://api.jina.ai/v1",
        rerank_model_id="jina-reranker-v3",
    )
    monkeypatch.setattr(rerank_module, "get_jina_config", lambda: bogus)
    with pytest.raises(RuntimeError, match="JINA_API_KEY"):
        jina_rerank("query", ["doc"])


def test_unreachable_server_names_the_fix(monkeypatch):
    """A dead endpoint fails with a diagnosis, not a raw socket errno."""
    rerank_module = importlib.import_module("ingestlib.foundations.llm.jina.rerank")
    dead = JinaConfig(
        api_key="jina_x",
        base_url="http://localhost:1/v1",
        rerank_model_id="jina-reranker-v3",
    )
    monkeypatch.setattr(rerank_module, "get_jina_config", lambda: dead)
    with pytest.raises(RuntimeError, match="could not reach"):
        jina_rerank("query", ["doc"])
