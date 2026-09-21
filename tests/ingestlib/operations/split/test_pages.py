"""Block extraction — pure, always run."""
from pathlib import Path

from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.parse.models import FigureImage, PageResult, ParseResult
from ingestlib.operations.split.pages import MAX_PAGES, _paragraphs, extract_split_pages


def _region(rtype, rid, text=""):
    return Region(region_type=rtype, bbox=BoundingBox(x=0, y=rid * 100, width=100, height=50),
                  region_id=rid, text=text, content=text)


def test_paragraphs_split_on_blank_lines():
    assert _paragraphs("one\n\ntwo\n\nthree") == ["one", "two", "three"]


def test_pdfium_style_text_without_blank_lines_regroups():
    text = "\r\n".join(f"line {i} " + "x" * 60 for i in range(60))  # no blank lines
    paras = _paragraphs(text)
    assert len(paras) > 1, "one giant block defeats the segmenter"


def test_caption_folds_into_preceding_visual():
    page = PageResult(
        page_num=1,
        regions=[
            _region("chart", 0, "chart data"),
            _region("figure_caption", 1, "Fig 1. Growth."),
            _region("text", 2, "Discussion paragraph."),
        ],
        figures=[FigureImage(region_id=0, region_type="chart", image_bytes=b"\x89PNG")],
    )
    pr = ParseResult(pages=[page], source_path=Path("x.pdf"), source_format="pdf")
    blocks = extract_split_pages(pr)[0].blocks
    assert len(blocks) == 2
    assert blocks[0].kind == "figure" and blocks[0].region_ids == (0, 1)
    assert "Fig 1. Growth." in blocks[0].markdown


def test_page_cap_at_500():
    pr = ParseResult(
        pages=[PageResult(page_num=i, text=f"p{i}") for i in range(1, 502)],
        source_path=Path("x.pdf"), source_format="pdf",
    )
    assert len(extract_split_pages(pr)) == MAX_PAGES


def test_caption_before_table_folds_forward():
    """Real documents put 'Table 1. ...' ABOVE the table — must stay atomic."""
    page = PageResult(
        page_num=1,
        regions=[
            _region("table_caption", 0, "Table 1. Demographics."),
            _region("table", 1, "<table><tr><td>a</td></tr></table>"),
            _region("text", 2, "Discussion paragraph."),
        ],
    )
    pr = ParseResult(pages=[page], source_path=Path("x.pdf"), source_format="pdf")
    blocks = extract_split_pages(pr)[0].blocks
    assert len(blocks) == 2
    assert blocks[0].kind == "table" and blocks[0].region_ids == (0, 1)
    assert blocks[0].markdown.index("Table 1.") < blocks[0].markdown.index("<table")


def test_orphan_caption_becomes_text_block():
    page = PageResult(
        page_num=1,
        regions=[
            _region("text", 0, "Paragraph."),
            _region("figure_caption", 1, "Orphan caption."),
            _region("text", 2, "Another paragraph."),
        ],
    )
    pr = ParseResult(pages=[page], source_path=Path("x.pdf"), source_format="pdf")
    blocks = extract_split_pages(pr)[0].blocks
    assert [b.kind for b in blocks] == ["text", "text", "text"]
