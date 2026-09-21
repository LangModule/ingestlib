"""Real verification of text embeddings against Nova 2 multimodal embeddings."""
import numpy as np
import pytest

from ingestlib.foundations.llm.bedrock.embedding import _invoke


_SHORT = "Invoice #12345 from Acme Corp, total $1,200.00 due 2026-08-15."


def test_envelope_shape_and_embedding_type():
    """Raw response has {embeddings: [{embeddingType: TEXT, embedding: [...]}]}."""
    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": 1024,
            "text": {"truncationMode": "END", "value": "hello world"},
        },
    }
    resp = _invoke(body)
    assert "embeddings" in resp
    assert len(resp["embeddings"]) == 1
    entry = resp["embeddings"][0]
    assert entry["embeddingType"] == "TEXT"
    assert isinstance(entry["embedding"], list)
    assert len(entry["embedding"]) == 1024


@pytest.mark.parametrize("dimension", [256, 384, 1024, 3072])
def test_supported_dimensions_return_exact_length(embed, dimension):
    vec = embed.text(_SHORT, dim=dimension)
    assert vec.shape == (dimension,)


def test_values_are_finite_and_not_all_zero(embed):
    vec = embed.text(_SHORT)
    assert np.all(np.isfinite(vec)), "vector must contain no NaN or Inf"
    assert np.any(np.abs(vec) > 1e-6), "vector must not be all zeros"
    assert np.max(np.abs(vec)) < 10.0, "values look out-of-range for normalized embeddings"


def test_purpose_is_not_a_noop(embed, cos_sim):
    """INDEX and RETRIEVAL of the same text must produce different vectors."""
    v_index = embed.text(_SHORT, purpose="GENERIC_INDEX")
    v_retrieval = embed.text(_SHORT, purpose="GENERIC_RETRIEVAL")
    sim = cos_sim(v_index, v_retrieval)
    assert sim < 0.9999, (
        f"INDEX and RETRIEVAL produced ~identical vectors (cos_sim={sim:.6f}); "
        "purpose parameter appears to be a no-op"
    )


def test_different_inputs_produce_different_vectors(embed, cos_sim):
    v_a = embed.text("The invoice for legal services is due next month.")
    v_b = embed.text("A calico cat naps in a patch of sunlight on the rug.")
    assert cos_sim(v_a, v_b) < 0.99


def test_semantic_related_closer_than_unrelated(embed, cos_sim):
    """The core RAG signal: semantically related texts must be closer than unrelated."""
    query = "invoice for goods sold with payment terms"
    related = "bill for items purchased, amount due in 30 days"
    unrelated = "a calico cat naps in a patch of sunlight on the rug"

    v_q = embed.text(query)
    v_related = embed.text(related)
    v_unrelated = embed.text(unrelated)

    sim_related = cos_sim(v_q, v_related)
    sim_unrelated = cos_sim(v_q, v_unrelated)

    assert sim_related > sim_unrelated, (
        f"related sim {sim_related:.4f} not greater than unrelated sim {sim_unrelated:.4f}"
    )


def test_long_text_auto_truncates(embed):
    """Long text (well over 8k tokens) should not error; server truncates via END mode."""
    long_ = "The quick brown fox jumps over the lazy dog. " * 400  # ~1800 tokens
    vec = embed.text(long_)
    assert vec.shape == (1024,)
    assert np.all(np.isfinite(vec))


def test_short_text_and_long_text_produce_different_vectors(embed, cos_sim):
    short = embed.text("hello")
    long_ = embed.text("The quick brown fox jumps over the lazy dog. " * 200)
    assert cos_sim(short, long_) < 0.99
