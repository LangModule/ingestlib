"""Fixtures for amazon.rerank-v1:0 tests: shared query/docs plus rate-limited callers."""
import threading
import time
from functools import lru_cache

import pytest

from ingestlib.foundations.llm import aws_rerank
from tests.foundations.llm._shared import (
    NLP_INDICES,
    NLP_QUERY,
    STANDARD_DOCS,
    UNRELATED_INDICES,
)

# amazon.rerank-v1:0 applied quota is 2 RPM = 30s between calls; 32s gives a safety margin.
_MIN_GAP_SECONDS = 32.0
_last_call_time = [0.0]
_call_lock = threading.Lock()


def _wait_for_rate_limit() -> None:
    with _call_lock:
        elapsed = time.time() - _last_call_time[0]
        if elapsed < _MIN_GAP_SECONDS:
            time.sleep(_MIN_GAP_SECONDS - elapsed)
        _last_call_time[0] = time.time()


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
    """Session-cached rerank; min-gap rate limiting kicks in only on cache miss."""

    @lru_cache(maxsize=None)
    def _cached(query: str, docs: tuple[str, ...], top_n: int | None):
        _wait_for_rate_limit()
        return aws_rerank(query, list(docs), top_n)

    def wrapper(query: str, docs: list[str], top_n: int | None = None):
        return _cached(query, tuple(docs), top_n)

    return wrapper


@pytest.fixture(scope="session")
def rate_limit_gate():
    """Expose the rate-limit gate for async tests that call aws_arerank directly."""
    return _wait_for_rate_limit
