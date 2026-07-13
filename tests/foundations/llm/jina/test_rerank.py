"""Real verification of jina_rerank() against jina-reranker-v3.

Skipped when JINA_API_KEY is not set (CI without secrets, fresh clones).
"""
import math
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("JINA_API_KEY"),
    reason="JINA_API_KEY not set in .env",
)


def test_returns_list_of_index_score_tuples(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs)
    assert isinstance(r, list)
    for item in r:
        assert isinstance(item, tuple) and len(item) == 2
        idx, score = item
        assert isinstance(idx, int)
        assert isinstance(score, float)


def test_all_indices_are_valid(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs)
    for idx, _ in r:
        assert 0 <= idx < len(standard_docs)


def test_indices_are_unique(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs)
    indices = [idx for idx, _ in r]
    assert len(indices) == len(set(indices))


def test_scores_are_finite(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs)
    for _, score in r:
        assert math.isfinite(score)


def test_scores_sorted_descending(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs)
    scores = [s for _, s in r]
    assert scores == sorted(scores, reverse=True)


def test_returns_all_docs_when_top_n_is_none(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs)
    assert len(r) == len(standard_docs)


def test_top_n_limits_result_count(cached_rerank, nlp_query, standard_docs):
    r = cached_rerank(nlp_query, standard_docs, top_n=3)
    assert len(r) == 3


def test_semantic_ordering_nlp_beats_unrelated(
    cached_rerank, nlp_query, standard_docs, nlp_indices, unrelated_indices,
):
    """The 3 NLP docs must all outrank the 3 unrelated docs by score."""
    r = cached_rerank(nlp_query, standard_docs)
    scores = dict(r)
    min_nlp = min(scores[i] for i in nlp_indices)
    max_unrelated = max(scores[i] for i in unrelated_indices)
    assert min_nlp > max_unrelated, (
        f"lowest NLP score {min_nlp:.6f} did not beat highest unrelated {max_unrelated:.6f}"
    )
