"""Client-side validation guards — no API calls, always run."""
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
