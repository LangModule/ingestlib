"""Real verification of OpenAI text embeddings.

Skipped when OPENAI_API_KEY is not set.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set in .env",
)

_SHORT = "Invoice #12345 from Acme Corp, total $1,200.00 due 2026-08-15."


@pytest.mark.parametrize("dimension", [256, 384, 1024])
def test_requested_dimensions_return_exact_length(oai, dimension):
    vec = oai.embed(_SHORT, dim=dimension)
    assert vec.shape == (dimension,)


def test_values_are_finite_and_not_all_zero(oai):
    vec = oai.embed(_SHORT)
    assert np.all(np.isfinite(vec)), "vector must contain no NaN or Inf"
    assert np.any(np.abs(vec) > 1e-6), "vector must not be all zeros"


def test_semantic_related_closer_than_unrelated(oai, cos_sim):
    """The core RAG signal must hold on this backend too."""
    v_q = oai.embed("invoice for goods sold with payment terms")
    v_related = oai.embed("bill for items purchased, amount due in 30 days")
    v_unrelated = oai.embed("a calico cat naps in a patch of sunlight on the rug")
    assert cos_sim(v_q, v_related) > cos_sim(v_q, v_unrelated)


def test_purpose_is_a_documented_noop():
    """Symmetric embeddings: INDEX and RETRIEVAL of the same text are identical."""
    from ingestlib.foundations.llm.openai import embed_text

    v_index = embed_text(_SHORT, purpose="GENERIC_INDEX")
    v_retrieval = embed_text(_SHORT, purpose="GENERIC_RETRIEVAL")
    assert np.allclose(np.asarray(v_index), np.asarray(v_retrieval), atol=1e-6)


async def test_aembed_text_matches_sync(oai):
    from ingestlib.foundations.llm.openai import aembed_text

    sync_vec = oai.embed(_SHORT)
    async_vec = np.asarray(await aembed_text(_SHORT), dtype=float)
    assert np.allclose(sync_vec, async_vec, atol=1e-6)


def test_embedder_cache_and_reset():
    from ingestlib.foundations.llm.openai import reset_embedders
    from ingestlib.foundations.llm.openai.embedding import _embedder

    a = _embedder(1024)
    assert _embedder(1024) is a
    assert _embedder(384) is not a, "each dimension gets its own instance"
    reset_embedders()
    assert _embedder(1024) is not a
