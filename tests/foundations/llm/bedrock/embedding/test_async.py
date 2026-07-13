"""Async wrappers must produce the same vectors as the sync primitives."""
import asyncio

import numpy as np

from ingestlib.foundations.llm.bedrock.embedding import (
    aembed_image,
    aembed_text,
    embed_image,
    embed_text,
)


async def test_aembed_text_matches_sync():
    text = "the quarterly earnings report for Q3 2026"
    sync_vec = np.asarray(embed_text(text), dtype=float)
    async_vec = np.asarray(await aembed_text(text), dtype=float)
    # deterministic API + same input = identical vectors
    assert np.allclose(sync_vec, async_vec, atol=1e-6)


async def test_aembed_image_matches_sync(photo_path):
    data = photo_path.read_bytes()
    sync_vec = np.asarray(embed_image(data, format="jpeg"), dtype=float)
    async_vec = np.asarray(await aembed_image(data, format="jpeg"), dtype=float)
    assert np.allclose(sync_vec, async_vec, atol=1e-6)


async def test_aembed_text_returns_expected_dimension():
    vec = await aembed_text("hello", dimension=384)
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(np.isfinite(v) for v in vec)


async def test_aembed_image_returns_expected_dimension(doc_text_path):
    vec = await aembed_image(
        doc_text_path.read_bytes(),
        format="png",
        detail_level="DOCUMENT_IMAGE",
        dimension=256,
    )
    assert isinstance(vec, list)
    assert len(vec) == 256
    assert all(np.isfinite(v) for v in vec)


async def test_async_gather_runs_concurrently(doc_text_path, doc_chart_path):
    """asyncio.gather over embed calls should complete without error and return distinct vectors."""
    text_v, doc_text_v, doc_chart_v = await asyncio.gather(
        aembed_text("a document image"),
        aembed_image(doc_text_path.read_bytes(), format="png", detail_level="DOCUMENT_IMAGE"),
        aembed_image(doc_chart_path.read_bytes(), format="png", detail_level="DOCUMENT_IMAGE"),
    )
    assert len(text_v) == 1024
    assert len(doc_text_v) == 1024
    assert len(doc_chart_v) == 1024
    # sanity — different content, different vectors
    assert not np.allclose(np.asarray(doc_text_v), np.asarray(doc_chart_v), atol=1e-4)
