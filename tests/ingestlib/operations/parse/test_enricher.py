"""Caption-linking geometry — pure logic, always run (Nova calls live in e2e tests)."""
from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.parse.enricher import find_caption


_PAGE_HEIGHT = 2000


def _region(rtype: str, y: float, rid: int = 0, text: str = "", height: float = 50) -> Region:
    return Region(
        region_type=rtype,  # type: ignore[arg-type]
        bbox=BoundingBox(x=100, y=y, width=800, height=height),
        region_id=rid,
        text=text,
    )


def test_caption_directly_below_is_linked():
    chart = _region("chart", y=500, height=400)
    caption = _region("figure_caption", y=920, text="Growth")
    assert find_caption(chart, [chart, caption], _PAGE_HEIGHT) == "Growth"


def test_caption_directly_above_is_linked():
    chart = _region("chart", y=500, height=400)
    caption = _region("figure_caption", y=430, text="Profitability")
    assert find_caption(chart, [chart, caption], _PAGE_HEIGHT) == "Profitability"


def test_nearest_of_two_captions_wins():
    chart = _region("chart", y=500, height=400)
    near = _region("figure_caption", y=910, rid=1, text="near")
    far = _region("figure_caption", y=430, rid=2, text="far")  # gap 20 above vs 10 below
    assert find_caption(chart, [chart, far, near], _PAGE_HEIGHT) == "near"


def test_caption_too_far_away_is_ignored():
    chart = _region("chart", y=100, height=200)
    caption = _region("figure_caption", y=1500, text="unrelated")  # 1200px gap
    assert find_caption(chart, [chart, caption], _PAGE_HEIGHT) == ""


def test_caption_without_horizontal_overlap_is_ignored():
    chart = _region("chart", y=500, height=400)
    caption = Region(
        region_type="figure_caption",
        bbox=BoundingBox(x=2000, y=910, width=300, height=50),  # far right column
        text="other column",
    )
    assert find_caption(chart, [chart, caption], _PAGE_HEIGHT) == ""


def test_non_caption_regions_never_match():
    chart = _region("chart", y=500, height=400)
    text = _region("text", y=910, text="body text right below")
    assert find_caption(chart, [chart, text], _PAGE_HEIGHT) == ""
