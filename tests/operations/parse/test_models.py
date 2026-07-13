"""ParseResult / PageResult / FigureImage behavior — pure logic, always run."""
from pathlib import Path

import pytest

from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.parse.models import FigureImage, PageResult, ParseResult


def _page(page_num: int = 1, **overrides) -> PageResult:
    defaults = dict(
        page_num=page_num,
        text="hello world",
        markdown="## hello\n\nworld",
        regions=[
            Region(
                region_type="text",
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
                region_id=0,
                text="hello world",
            )
        ],
    )
    defaults.update(overrides)
    return PageResult(**defaults)


def _result(pages: list[PageResult]) -> ParseResult:
    return ParseResult(pages=pages, source_path=Path("doc.pdf"), source_format="pdf")


def test_figure_filename_matches_markdown_reference_convention():
    fig = FigureImage(region_id=7, region_type="chart", image_bytes=b"\x89PNG")
    assert fig.filename(page_num=2) == "page2_region7_chart.png"


def test_word_count_and_has_native_text():
    p = _page(native_text="native layer")
    assert p.word_count == 2
    assert p.has_native_text is True
    assert _page(native_text="  ").has_native_text is False


def test_region_by_id_found_and_missing():
    p = _page()
    assert p.region_by_id(0).text == "hello world"
    with pytest.raises(IndexError, match="region_id=99"):
        p.region_by_id(99)


def test_page_by_num_found_and_missing():
    r = _result([_page(1), _page(2)])
    assert r.page_by_num(2).page_num == 2
    with pytest.raises(IndexError, match="page_num=5"):
        r.page_by_num(5)


def test_document_markdown_joins_pages_in_order():
    r = _result([
        _page(1, markdown="page one"),
        _page(2, markdown="page two"),
    ])
    assert r.markdown.index("page one") < r.markdown.index("page two")


def test_total_word_count_sums_pages():
    r = _result([_page(1, text="a b c"), _page(2, text="d e")])
    assert r.total_word_count == 5


def test_save_images_writes_figures_with_canonical_names(tmp_path):
    fig = FigureImage(region_id=3, region_type="figure", image_bytes=b"\x89PNGdata")
    r = _result([_page(1, figures=[fig])])
    written = r.save_images(tmp_path)
    assert written == [tmp_path / "page1_region3_figure.png"]
    assert written[0].read_bytes() == b"\x89PNGdata"


def test_results_are_frozen():
    r = _result([_page()])
    with pytest.raises(Exception):
        r.source_format = "docx"  # type: ignore[misc]
