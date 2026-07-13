"""Async wrapper arerank() produces same-shape results as sync.

Skipped by default — see test_rerank.py for the reason.
"""
import os

import pytest

from ingestlib.foundations.llm import aws_arerank

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AWS_RERANK") != "1",
    reason="AWS rerank on 2 RPM quota; set RUN_AWS_RERANK=1 to opt in",
)


async def test_arerank_returns_correct_structure(nlp_query, standard_docs, rate_limit_gate):
    rate_limit_gate()
    r = await aws_arerank(nlp_query, standard_docs)
    assert isinstance(r, list)
    for item in r:
        assert isinstance(item, tuple) and len(item) == 2
        idx, score = item
        assert isinstance(idx, int) and 0 <= idx < len(standard_docs)
        assert isinstance(score, float)
