"""Real DOCX and PPTX through the real LibreOffice conversion — the office
loader's first encounter with actual Office files (until now only its error
paths were tested). Skips cleanly where LibreOffice isn't installed."""
import shutil
from pathlib import Path

import pytest

from ingestlib.operations.parse.loaders import load_office, load_office_content

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_DOCX = _TESTS_DIR / "data" / "word" / "project-plan.docx"
_PPTX = _TESTS_DIR / "data" / "ppt" / "slide-deck.pptx"

pytestmark = pytest.mark.skipif(
    shutil.which("soffice") is None,
    reason="LibreOffice (soffice) is not installed — DOCX/PPTX need it",
)


def test_docx_converts_to_rendered_pages():
    pages, _ = load_office(_DOCX)
    assert len(pages) >= 1
    assert all(p.image_bytes.startswith(b"\x89PNG") for p in pages)
    # A born-digital document's text must survive the conversion to PDF.
    assert any(p.native_text.strip() for p in pages)


def test_pptx_converts_one_page_per_slide():
    pages, _ = load_office(_PPTX)
    assert len(pages) > 1, "a slide deck must yield multiple pages"
    assert all(p.image_bytes.startswith(b"\x89PNG") for p in pages)


def test_docx_content_shape_extracts_text():
    """The cheap path classify/split use — native text without OCR."""
    pages, _ = load_office_content(_DOCX)
    assert any(p.text.strip() for p in pages)
