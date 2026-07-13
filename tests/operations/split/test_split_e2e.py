"""Real split against Bedrock Nova. Opt-in via RUN_SPLIT_E2E=1."""
import os
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SPLIT_E2E") != "1",
    reason="split e2e is opt-in: set RUN_SPLIT_E2E=1 (needs Bedrock access)",
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
    for c in result.chunks:
        assert c.token_estimate <= 1024 or c.kind in ("table", "figure")
