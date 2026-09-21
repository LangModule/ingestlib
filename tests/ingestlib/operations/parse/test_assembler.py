"""Markdown/text assembly from regions — pure logic, always run."""
from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.parse.assembler import assemble_markdown, assemble_text, render_region
from ingestlib.operations.parse.models import FigureImage


def _region(rtype: str, rid: int = 0, text: str = "", content: str = "") -> Region:
    return Region(
        region_type=rtype,  # type: ignore[arg-type]
        bbox=BoundingBox(x=0, y=float(rid * 100), width=100, height=50),
        region_id=rid,
        text=text,
        content=content or text,
    )


def test_multiline_title_collapses_to_one_heading_line():
    md = render_region(_region("title", text="Uber is best positioned\nto capitalize"), {}, 1)
    assert md == "## Uber is best positioned to capitalize"
    assert "\n" not in md


def test_header_footer_excluded_from_markdown_and_text():
    regions = [
        _region("header", 0, text="PLOS MEDICINE"),
        _region("text", 1, text="Actual content."),
        _region("footer", 2, text="page 7"),
    ]
    md = assemble_markdown(regions, [], page_num=1)
    txt = assemble_text(regions)
    assert "PLOS MEDICINE" not in md and "page 7" not in md
    assert "PLOS MEDICINE" not in txt and "page 7" not in txt
    assert "Actual content." in md and "Actual content." in txt


def test_table_renders_raw_html_content():
    html = "<table><tr><td>a</td></tr></table>"
    md = render_region(_region("table", content=html), {}, 1)
    assert md == html


def test_chart_renders_image_ref_plus_data_table():
    fig = FigureImage(
        region_id=5, region_type="chart", image_bytes=b"\x89PNG",
        caption="Growth", description="| Year | ARR |\n| 2022 | $56B |",
    )
    region = _region("chart", rid=5, content=fig.description)
    md = render_region(region, {5: fig}, page_num=2)
    assert "![Growth](page2_region5_chart.png)" in md
    assert "| 2022 | $56B |" in md


def test_figure_renders_image_ref_plus_blockquoted_description():
    fig = FigureImage(
        region_id=3, region_type="figure", image_bytes=b"\x89PNG",
        caption="", description="A diagram of consumers and merchants.",
    )
    md = render_region(_region("figure", rid=3), {3: fig}, page_num=1)
    assert "![figure](page1_region3_figure.png)" in md
    assert "> A diagram of consumers and merchants." in md


def test_bare_latex_formula_gets_display_block():
    md = render_region(_region("formula", content=r"E = mc^2"), {}, 1)
    assert md.startswith("$$") and md.endswith("$$")


def test_already_delimited_formula_left_alone():
    md = render_region(_region("formula", content=r"$E = mc^2$"), {}, 1)
    assert md == r"$E = mc^2$"


def test_captions_render_italic():
    md = render_region(_region("figure_caption", text="Fig 1. Flowchart."), {}, 1)
    assert md == "*Fig 1. Flowchart.*"


def test_assemble_markdown_preserves_reading_order():
    regions = [
        _region("title", 0, text="Heading"),
        _region("text", 1, text="First paragraph."),
        _region("text", 2, text="Second paragraph."),
    ]
    md = assemble_markdown(regions, [], page_num=1)
    assert md.index("Heading") < md.index("First paragraph.") < md.index("Second paragraph.")
