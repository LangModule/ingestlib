"""Source preparation for extract — markers, windows, native path. Pure/real."""
from pathlib import Path

import pytest

from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.extract.context import (
    MANY_WINDOW_PAGES,
    MAX_PAGES,
    SourcePage,
    build_pages,
    format_window,
    prepare_windows,
)
from ingestlib.operations.parse.models import PageResult, ParseResult

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent


def _region(rid: int, text: str, rtype: str = "text", content: str = "") -> Region:
    return Region(
        region_type=rtype,  # type: ignore[arg-type]
        bbox=BoundingBox(x=0, y=rid * 10, width=100, height=9),
        region_id=rid,
        text=text,
        content=content,
    )


def _parse_result(pages: list[PageResult]) -> ParseResult:
    return ParseResult(pages=pages, source_path=Path("doc.pdf"), source_format="pdf")


def test_parse_pages_get_self_citing_markers():
    pr = _parse_result([
        PageResult(page_num=4, regions=[
            _region(0, "Apple Inc."),
            _region(2, "raw", rtype="table", content="<table>383,285</table>"),
            _region(3, ""),  # empty — must be skipped
        ]),
    ])
    pages = build_pages(pr)
    assert len(pages) == 1
    page = pages[0]
    assert page.page_num == 4
    assert "[p4:r0] Apple Inc." in page.text
    # content wins over text when present (tables carry their HTML)
    assert "[p4:r2] <table>383,285</table>" in page.text
    assert page.region_ids == frozenset({0, 2})
    assert page.region_texts[2] == "<table>383,285</table>"


def test_native_path_has_page_level_provenance_only():
    pages = build_pages(_TESTS_DIR / "data" / "pdf" / "finance-10k.pdf")
    assert pages, "the fixture has pages"
    assert all(p.region_ids is None for p in pages)
    assert any("383,285" in p.text for p in pages), "native text layer present"


def test_many_mode_windows_overlap_by_one_page():
    pages = [
        SourcePage(page_num=i, text=f"page {i}", region_ids=frozenset(),
                   region_texts={}, images=[])
        for i in range(1, 11)
    ]
    windows, used = prepare_windows(pages, "many", None)
    assert used == 10
    assert all(len(w) <= MANY_WINDOW_PAGES for w in windows)
    # consecutive windows share exactly one page
    for a, b in zip(windows, windows[1:]):
        assert a[-1].page_num == b[0].page_num
    covered = {p.page_num for w in windows for p in w}
    assert covered == set(range(1, 11)), "every page appears in some window"


def test_many_mode_drops_fully_contained_trailing_window():
    pages = [
        SourcePage(page_num=i, text="x", region_ids=frozenset(), region_texts={}, images=[])
        for i in range(1, 5)   # exactly one window of 4; stride would add a stub
    ]
    windows, _ = prepare_windows(pages, "many", None)
    assert len(windows) == 1


def test_one_mode_single_window_up_to_twenty_pages():
    pages = [
        SourcePage(page_num=i, text="x", region_ids=frozenset(), region_texts={}, images=[])
        for i in range(1, 21)
    ]
    windows, _ = prepare_windows(pages, "one", None)
    assert len(windows) == 1

    pages_25 = pages + [
        SourcePage(page_num=i, text="x", region_ids=frozenset(), region_texts={}, images=[])
        for i in range(21, 26)
    ]
    windows, _ = prepare_windows(pages_25, "one", None)
    assert [len(w) for w in windows] == [20, 5]


def test_target_pages_selects_before_windowing():
    pages = [
        SourcePage(page_num=i, text=f"page {i}", region_ids=frozenset(),
                   region_texts={}, images=[])
        for i in range(1, 11)
    ]
    windows, used = prepare_windows(pages, "one", "2,5-6")
    assert used == 3
    assert [p.page_num for p in windows[0]] == [2, 5, 6]


def test_hundred_page_cap_applies():
    pages = [
        SourcePage(page_num=i, text="x", region_ids=frozenset(), region_texts={}, images=[])
        for i in range(1, 131)
    ]
    _, used = prepare_windows(pages, "one", None)
    assert used == MAX_PAGES


def test_format_window_delimits_pages():
    pages = [
        SourcePage(page_num=1, text="[p1:r0] hello", region_ids=frozenset({0}),
                   region_texts={0: "hello"}, images=[]),
        SourcePage(page_num=2, text="", region_ids=frozenset(), region_texts={}, images=[]),
    ]
    body = format_window(pages)
    assert "--- page 1 ---" in body and "[p1:r0] hello" in body
    assert "--- page 2 ---" in body and "no extractable text" in body


def test_bad_target_pages_raises():
    pages = [
        SourcePage(page_num=1, text="x", region_ids=frozenset(), region_texts={}, images=[])
    ]
    with pytest.raises(ValueError, match="target_pages"):
        prepare_windows(pages, "one", "nope")
