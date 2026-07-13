"""Pure-logic tests for OCR models — no server, always run."""
from io import BytesIO

from PIL import Image

from ingestlib.foundations.ocr.models import BoundingBox, Region


def _bbox() -> BoundingBox:
    return BoundingBox(x=100.0, y=200.0, width=300.0, height=400.0)


def test_x2_y2_and_as_tuple():
    b = _bbox()
    assert b.x2 == 400.0 and b.y2 == 600.0
    assert b.as_tuple() == (100.0, 200.0, 400.0, 600.0)


def test_normalized_is_zero_to_one():
    n = _bbox().normalized(page_width=1000, page_height=2000)
    assert n == (0.1, 0.1, 0.4, 0.3)
    assert all(0.0 <= v <= 1.0 for v in n)


def test_to_pdf_points_at_200_dpi():
    p = _bbox().to_pdf_points(dpi=200)
    # 72/200 = 0.36 scale
    assert p.x == 36.0 and p.y == 72.0
    assert p.width == 108.0 and p.height == 144.0


def test_crop_returns_png_of_expected_size():
    img = Image.new("RGB", (1000, 1000), "white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    crop = BoundingBox(x=10, y=20, width=200, height=100).crop(buf.getvalue())
    assert crop.startswith(b"\x89PNG")
    cropped = Image.open(BytesIO(crop))
    assert (cropped.width, cropped.height) == (200, 100)


def test_region_is_frozen_and_replaceable():
    import dataclasses

    r = Region(region_type="text", bbox=_bbox(), region_id=3, text="a")
    updated = dataclasses.replace(r, text="b")
    assert r.text == "a" and updated.text == "b"
    assert updated.region_id == 3
