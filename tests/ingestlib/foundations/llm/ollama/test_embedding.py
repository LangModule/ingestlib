"""Real verification of Ollama text embeddings.

Opt-in via RUN_OLLAMA_E2E=1: needs a local Ollama serving the configured
embedding model.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_E2E") != "1",
    reason="ollama e2e is opt-in: set RUN_OLLAMA_E2E=1 (needs a local Ollama "
           "with the configured models pulled)",
)

_SHORT = "Invoice #12345 from Acme Corp, total $1,200.00 due 2026-08-15."


def test_native_dimension_returns_exact_length(olm):
    vec = olm.embed(_SHORT, dim=1024)
    assert vec.shape == (1024,)


def test_mismatched_dimension_raises():
    """The server returns the model's native size — a wrong request must fail
    loudly, never index vectors of a surprise length."""
    from ingestlib.foundations.llm.ollama import embed_text

    with pytest.raises(ValueError, match="native"):
        embed_text(_SHORT, dimension=384)


def test_values_are_finite_and_not_all_zero(olm):
    vec = olm.embed(_SHORT)
    assert np.all(np.isfinite(vec)), "vector must contain no NaN or Inf"
    assert np.any(np.abs(vec) > 1e-6), "vector must not be all zeros"


def test_semantic_related_closer_than_unrelated(olm, cos_sim):
    """The core RAG signal must hold on this backend too."""
    v_q = olm.embed("invoice for goods sold with payment terms")
    v_related = olm.embed("bill for items purchased, amount due in 30 days")
    v_unrelated = olm.embed("a calico cat naps in a patch of sunlight on the rug")
    assert cos_sim(v_q, v_related) > cos_sim(v_q, v_unrelated)


def test_purpose_is_a_documented_noop():
    """Symmetric embeddings: INDEX and RETRIEVAL of the same text are the same
    vector. Local inference is deterministic in practice, but separate calls
    keep the same tolerance as the cloud backends."""
    from ingestlib.foundations.llm.ollama import embed_text

    v_index = embed_text(_SHORT, purpose="GENERIC_INDEX")
    v_retrieval = embed_text(_SHORT, purpose="GENERIC_RETRIEVAL")
    assert np.allclose(np.asarray(v_index), np.asarray(v_retrieval), atol=1e-3)


async def test_aembed_text_matches_sync(olm):
    from ingestlib.foundations.llm.ollama import aembed_text

    sync_vec = olm.embed(_SHORT)
    async_vec = np.asarray(await aembed_text(_SHORT), dtype=float)
    assert np.allclose(sync_vec, async_vec, atol=1e-3)


def test_embedder_cache_and_reset():
    from ingestlib.foundations.llm.ollama import reset_embedders
    from ingestlib.foundations.llm.ollama.embedding import _get_embedder

    a = _get_embedder()
    assert _get_embedder() is a
    reset_embedders()
    assert _get_embedder() is not a
