"""Standalone split without a text layer must say so, never emit empty chunks."""
from pathlib import Path

import pytest

from ingestlib.operations.split.pages import extract_split_pages

_TEXTLESS_PDF = (
    b"%PDF-1.1\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
)

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent


def test_textless_pdf_points_at_parse(tmp_path):
    pdf = tmp_path / "blank-scan.pdf"
    pdf.write_bytes(_TEXTLESS_PDF)
    with pytest.raises(ValueError, match="Run parse\\(\\) first"):
        extract_split_pages(pdf)


def test_image_input_points_at_parse():
    """Images have no text layer — split standalone must route users to parse()."""
    photo = _TESTS_DIR / "data" / "images" / "photo.jpg"
    with pytest.raises(ValueError, match="Run parse\\(\\) first"):
        extract_split_pages(photo)
