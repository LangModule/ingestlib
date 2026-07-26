"""Standalone classify on a document with nothing to read must say so."""
import pytest

# A valid single-page PDF with no text layer and no images — a "blank scan".
_TEXTLESS_PDF = (
    b"%PDF-1.1\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n"
)


def test_textless_imageless_pdf_points_at_parse(tmp_path):
    from ingestlib.operations.classify.chunker import extract_pages

    pdf = tmp_path / "blank-scan.pdf"
    pdf.write_bytes(_TEXTLESS_PDF)
    with pytest.raises(ValueError, match="Run parse\\(\\) first"):
        extract_pages(pdf)
