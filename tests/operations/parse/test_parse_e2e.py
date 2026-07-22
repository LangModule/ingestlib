"""End-to-end parse smoke tests against the real stack (VL server + the
configured LLM provider).

Opt-in via RUN_PARSE_E2E=1 — a full document parse costs ~30-60s and a few
tenths of a cent in LLM calls.
"""
import os
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_DATA_DIR = _TESTS_DIR / "data"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PARSE_E2E") != "1",
    reason="parse e2e is opt-in: set RUN_PARSE_E2E=1 (needs VL server + LLM-provider access)",
)


@pytest.fixture(scope="session")
def result():
    """One real end-to-end parse, shared across every test in this module."""
    from ingestlib.operations.parse import parse

    return parse(_DATA_DIR / "pdf" / "uber-earnings.pdf")


def test_all_pages_parsed(result):
    assert result.page_count == 3
    assert [p.page_num for p in result.pages] == [1, 2, 3]


def test_markdown_produced_per_page(result):
    assert all(p.markdown.strip() for p in result.pages)


def test_figures_extracted_with_images(result):
    figures = [f for p in result.pages for f in p.figures]
    assert figures, "expected chart/figure extractions from the earnings deck"
    assert all(f.image_bytes.startswith(b"\x89PNG") for f in figures)
    assert all(f.description.strip() for f in figures)


def test_charts_became_data_tables(result):
    chart_regions = [
        r for p in result.pages for r in p.regions if r.region_type == "chart"
    ]
    assert chart_regions, "the earnings deck contains charts"
    # LLM-enriched chart content is a markdown table (or a description for diagrams)
    assert any("|" in r.content for r in chart_regions)


def test_markdown_references_match_figure_filenames(result):
    for p in result.pages:
        for fig in p.figures:
            assert fig.filename(p.page_num) in p.markdown


def test_page_furniture_excluded_from_text(result):
    for p in result.pages:
        furniture = [r for r in p.regions if r.region_type in ("header", "footer")]
        for r in furniture:
            if r.text.strip():
                assert r.text not in p.text


def test_checksum_and_metadata(result):
    assert result.source_checksum and len(result.source_checksum) == 64
    assert result.source_format == "pdf"
    assert result.parse_duration_seconds > 0


def test_save_images_roundtrip(result, tmp_path):
    written = result.save_images(tmp_path)
    assert len(written) == sum(len(p.figures) for p in result.pages)
    assert all(w.exists() and w.stat().st_size > 0 for w in written)
