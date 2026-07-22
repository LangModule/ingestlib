"""Real split against the configured LLM provider. Opt-in via RUN_SPLIT_E2E=1."""
import os
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SPLIT_E2E") != "1",
    reason="split e2e is opt-in: set RUN_SPLIT_E2E=1 (needs LLM-provider access)",
)


@pytest.fixture(scope="session")
def result():
    """One real end-to-end split, shared across every test in this module."""
    from ingestlib.operations.split import split

    return split(_TESTS_DIR / "data" / "pdf" / "clinical-study.pdf", category="research_paper")


def test_sections_cover_all_pages_in_order(result):
    covered = [p for s in result.sections for p in s.pages]
    assert covered == [1, 2, 3, 4]
    assert result.pages_used == 4


def test_chunk_ids_sequential(result):
    assert [c.chunk_id for c in result.chunks] == list(range(len(result.chunks)))


def test_embedding_text_carries_breadcrumb(result):
    for c in result.chunks:
        first = c.embedding_text.splitlines()[0]
        assert first.startswith("[research_paper › ")
        assert c.section in first


def test_no_layout_named_sections(result):
    for name in result.section_names:
        assert name not in ("tables", "figures", "images", "text")


def test_chunks_respect_max_tokens_or_are_single_block(result):
    from ingestlib.operations.split.splitter import DEFAULT_MAX_CHUNK_TOKENS

    for c in result.chunks:
        assert c.token_estimate <= DEFAULT_MAX_CHUNK_TOKENS or c.kind in ("table", "figure")


def test_user_vocabulary_constrains_sections():
    """Live closed-set split: sections come from MY names (or `other`),
    never from Pass 1 — which does not run at all."""
    from ingestlib.operations.split import split

    vocab = {
        "study_design": "Objectives, methods, participants, and procedures",
        "findings": "Results, outcomes, statistics, and discussion",
    }
    result = split(
        _TESTS_DIR / "data" / "pdf" / "clinical-study.pdf",
        vocabulary=vocab, category="research_paper",
    )
    assert set(result.section_names) <= set(vocab) | {"other"}
    assert result.chunks, "user-vocabulary sections still chunk"
    for c in result.chunks:
        assert c.embedding_text.startswith("[research_paper › ")


def test_semantic_subsplit_respects_budget_and_covers_everything():
    """Live sub-split: an oversized multi-topic group must come back as
    budget-fitting, fully-covering sub-chunks with their own headings."""
    import asyncio

    from ingestlib.operations.split.pages import Block
    from ingestlib.operations.split.segmenter import _enforce_budget

    def para(topic: str, i: int) -> Block:
        text = f"{topic} paragraph {i}. " + ("It details the procedure and results. " * 20)
        return Block(page_num=1, kind="text", markdown=text, text=text, region_ids=())

    # ~2 topics × 4 paragraphs × ~190 tokens ≈ 1500 tokens — over a 768 budget.
    blocks = [para("Participant recruitment and consent", i) for i in range(4)] + [
        para("Statistical analysis and dropout handling", i) for i in range(4)
    ]
    groups = asyncio.run(
        _enforce_budget(
            "methods", [(list(range(len(blocks))), "methods overview")],
            blocks, 768, asyncio.Semaphore(2),
        )
    )
    covered = sorted(i for g in groups for i in g[0])
    assert covered == list(range(len(blocks))), "every block exactly once"
    for indexes, _heading in groups:
        assert sum(blocks[i].tokens for i in indexes) <= 768 or len(indexes) == 1
    assert len(groups) >= 2
