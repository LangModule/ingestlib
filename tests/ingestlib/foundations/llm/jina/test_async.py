"""Async wrapper jina_arerank() produces same-shape results as sync."""
import os

import pytest

from ingestlib.foundations.llm import jina_arerank

pytestmark = pytest.mark.skipif(
    not os.getenv("JINA_API_KEY"),
    reason="JINA_API_KEY not set in .env",
)


async def test_arerank_returns_correct_structure(nlp_query, standard_docs):
    r = await jina_arerank(nlp_query, standard_docs)
    assert isinstance(r, list)
    for item in r:
        assert isinstance(item, tuple) and len(item) == 2
        idx, score = item
        assert isinstance(idx, int) and 0 <= idx < len(standard_docs)
        assert isinstance(score, float)
