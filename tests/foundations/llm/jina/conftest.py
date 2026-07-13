"""Fixtures for Jina reranker tests: shared query/docs plus session-cached rerank calls.

Jina's free tier is 100 RPM — no rate limiting needed for a handful of test calls.
"""
from functools import lru_cache

import pytest

from ingestlib.foundations.llm import jina_rerank
from tests.foundations.llm._shared import (
    NLP_INDICES,
    NLP_QUERY,
    STANDARD_DOCS,
    UNRELATED_INDICES,
)


@pytest.fixture(scope="session")
def nlp_query() -> str:
    return NLP_QUERY


@pytest.fixture(scope="session")
def standard_docs() -> list[str]:
    return list(STANDARD_DOCS)


@pytest.fixture(scope="session")
def nlp_indices() -> frozenset[int]:
    return NLP_INDICES


@pytest.fixture(scope="session")
def unrelated_indices() -> frozenset[int]:
    return UNRELATED_INDICES


@pytest.fixture(scope="session")
def cached_rerank():
    """Session-cached rerank — same result reused across the eight structural checks."""

    @lru_cache(maxsize=None)
    def _cached(query: str, docs: tuple[str, ...], top_n: int | None):
        return jina_rerank(query, list(docs), top_n)

    def wrapper(query: str, docs: list[str], top_n: int | None = None):
        return _cached(query, tuple(docs), top_n)

    return wrapper
