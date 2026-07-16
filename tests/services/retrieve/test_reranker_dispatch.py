"""Reranker selection via config.yaml's `reranker` key — pure, always run.

An unknown value must fail loudly before any embedding or store call, the
same contract default_store() enforces for `vector_store`. Known values are
proven to pass the guard by letting execution reach the embed boundary and
stopping it there with a sentinel — no network, no faked responses. The
jina/aws/none behavioral paths need real embeddings and live in the services
e2e suite.
"""
import asyncio
import importlib

import pytest

import ingestlib.config as config_module
from ingestlib.services import aretrieve
from ingestlib.storage import SqliteStore

# services/__init__.py re-exports `retrieve` as a function, shadowing the
# subpackage on attribute lookup; grab the module object explicitly.
retriever_module = importlib.import_module("ingestlib.services.retrieve.retriever")


class _ReachedEmbed(Exception):
    """Raised by the patched embed boundary — reaching it means every guard passed."""


async def _stop_at_embed(*args, **kwargs):
    raise _ReachedEmbed


def _with_reranker(monkeypatch, name: str) -> None:
    current = config_module.get_config()  # materialize the lazy singleton
    patched = current.__class__(**{**current.__dict__, "reranker": name})
    monkeypatch.setattr(config_module, "_config", patched)


def test_unknown_reranker_raises_with_choices(monkeypatch):
    _with_reranker(monkeypatch, "cohere")
    # SqliteStore() never connects, and the guard fires before the embed call,
    # so this raises without touching any network or database.
    with pytest.raises(ValueError, match="aws.*jina.*none"):
        asyncio.run(aretrieve("a question", store=SqliteStore()))


@pytest.mark.parametrize("name", ["jina", "aws", "none"])
def test_known_reranker_values_pass_the_guard(monkeypatch, name):
    _with_reranker(monkeypatch, name)
    monkeypatch.setattr(retriever_module, "aembed_text", _stop_at_embed)
    with pytest.raises(_ReachedEmbed):
        asyncio.run(aretrieve("a question", store=SqliteStore()))
