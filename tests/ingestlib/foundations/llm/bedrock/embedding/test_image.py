"""Real verification of image embeddings against Nova 2 multimodal embeddings."""
import base64

import numpy as np
import pytest

from ingestlib.foundations.llm.bedrock.embedding import _invoke


def test_envelope_shape_and_embedding_type(photo_path):
    """Raw response has {embeddings: [{embeddingType: IMAGE, embedding: [...]}]}."""
    image_b64 = base64.b64encode(photo_path.read_bytes()).decode()
    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": 1024,
            "image": {
                "format": "jpeg",
                "detailLevel": "STANDARD_IMAGE",
                "source": {"bytes": image_b64},
            },
        },
    }
    resp = _invoke(body)
    assert "embeddings" in resp
    assert len(resp["embeddings"]) == 1
    entry = resp["embeddings"][0]
    assert entry["embeddingType"] == "IMAGE"
    assert isinstance(entry["embedding"], list)
    assert len(entry["embedding"]) == 1024


@pytest.mark.parametrize("dimension", [256, 384, 1024, 3072])
def test_supported_dimensions_return_exact_length(embed, photo_path, dimension):
    vec = embed.image(photo_path, dim=dimension)
    assert vec.shape == (dimension,)


def test_values_are_finite_and_not_all_zero(embed, photo_path):
    vec = embed.image(photo_path)
    assert np.all(np.isfinite(vec)), "vector must contain no NaN or Inf"
    assert np.any(np.abs(vec) > 1e-6), "vector must not be all zeros"
    assert np.max(np.abs(vec)) < 10.0, "values look out-of-range for normalized embeddings"


def test_detail_level_is_not_a_noop(embed, cos_sim, doc_text_path):
    """STANDARD_IMAGE and DOCUMENT_IMAGE of the same image must produce different vectors."""
    v_std = embed.image(doc_text_path, detail_level="STANDARD_IMAGE")
    v_doc = embed.image(doc_text_path, detail_level="DOCUMENT_IMAGE")
    sim = cos_sim(v_std, v_doc)
    assert sim < 0.9999, (
        f"STANDARD_IMAGE and DOCUMENT_IMAGE produced ~identical vectors (cos_sim={sim:.6f}); "
        "detail_level parameter appears to be a no-op"
    )


def test_different_images_produce_different_vectors(embed, cos_sim, photo_path, doc_chart_path):
    v_photo = embed.image(photo_path, detail_level="STANDARD_IMAGE")
    v_chart = embed.image(doc_chart_path, detail_level="DOCUMENT_IMAGE")
    assert cos_sim(v_photo, v_chart) < 0.99


# NOTE: an image↔image "documents cluster together" test used to live here.
# Measured against real fixtures the premise is fixture-dependent (three
# unrelated document images embed near-orthogonally, ~0.08-0.11 cosine), and
# nothing in the pipeline compares image embeddings to image embeddings —
# multimodal retrieval is text↔image, pinned in test_multimodal.py with
# wide measured margins.
