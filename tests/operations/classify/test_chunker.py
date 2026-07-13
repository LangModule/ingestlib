"""Chunking and page-extraction logic — pure, always run."""
from pathlib import Path

from ingestlib.operations.classify.chunker import (
    CHUNK_PAGES,
    MAX_PAGES,
    PAGE_TEXT_LIMIT,
    PageContent,
    cap_and_chunk,
    extract_pages,
)
from ingestlib.operations.parse.models import PageResult, ParseResult


def _pages(n: int) -> list[PageContent]:
    return [PageContent(text=f"page {i}", images=[]) for i in range(n)]


def test_small_doc_is_single_chunk():
    chunks, used = cap_and_chunk(_pages(5))
    assert len(chunks) == 1 and used == 5


def test_exactly_chunk_size_is_single_chunk():
    chunks, used = cap_and_chunk(_pages(CHUNK_PAGES))
    assert len(chunks) == 1 and used == CHUNK_PAGES


def test_large_doc_splits_into_20_page_chunks():
    chunks, used = cap_and_chunk(_pages(45))
    assert [len(c) for c in chunks] == [20, 20, 5]
    assert used == 45


def test_hard_cap_at_100_pages():
    chunks, used = cap_and_chunk(_pages(250))
    assert used == MAX_PAGES
    assert sum(len(c) for c in chunks) == MAX_PAGES
    assert len(chunks) == 5


def test_extract_pages_from_parse_result_prefers_markdown():
    pr = ParseResult(
        pages=[PageResult(page_num=1, text="plain", markdown="# rich", native_text="native")],
        source_path=Path("x.pdf"),
        source_format="pdf",
    )
    pages = extract_pages(pr)
    assert pages[0].text == "# rich"


def test_extract_pages_from_parse_result_uses_figure_crops():
    from ingestlib.operations.parse.models import FigureImage

    fig = FigureImage(region_id=1, region_type="chart", image_bytes=b"\x89PNGxx")
    pr = ParseResult(
        pages=[PageResult(page_num=1, markdown="text", figures=[fig])],
        source_path=Path("x.pdf"),
        source_format="pdf",
    )
    pages = extract_pages(pr)
    assert pages[0].images == [b"\x89PNGxx"]


def test_extract_pages_falls_back_to_text_then_native():
    pr = ParseResult(
        pages=[
            PageResult(page_num=1, text="plain only"),
            PageResult(page_num=2, native_text="native only"),
        ],
        source_path=Path("x.pdf"),
        source_format="pdf",
    )
    pages = extract_pages(pr)
    assert pages[0].text == "plain only"
    assert pages[1].text == "native only"


def test_extract_pages_trims_to_text_limit():
    pr = ParseResult(
        pages=[PageResult(page_num=1, markdown="x" * (PAGE_TEXT_LIMIT + 500))],
        source_path=Path("x.pdf"),
        source_format="pdf",
    )
    assert len(extract_pages(pr)[0].text) == PAGE_TEXT_LIMIT
