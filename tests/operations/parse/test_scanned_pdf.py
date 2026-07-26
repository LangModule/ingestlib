"""expenses.pdf — a real 16-page scan compilation of receipts, and the WORST
kind of scan: most pages carry a garbage text layer (bad scanner OCR —
mojibake, mirrored glyphs, rotated content) and two pages have none at all.

Parse's core contract — OCR the RENDER, never trust the text layer — is
exactly what this fixture exercises. The loader truths run always; the OCR
proof needs the real stack (RUN_PARSE_E2E=1) and parses a two-page excerpt
so it stays fast.
"""
import os
from pathlib import Path

import pytest

from ingestlib.operations.parse.loaders import load_pdf

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_SCAN = _TESTS_DIR / "data" / "pdf" / "expenses.pdf"


def test_scan_loads_with_its_untrustworthy_text_layer():
    pages, _ = load_pdf(_SCAN)
    assert len(pages) == 16
    assert all(p.image_bytes.startswith(b"\x89PNG") for p in pages)
    # Two pages have NO text layer at all — the pure-image case.
    empty = [i + 1 for i, p in enumerate(pages) if not p.native_text.strip()]
    assert 10 in empty and 13 in empty
    # The rest carry text, but it's scanner garbage — the layer exists and
    # still can't be trusted (why parse OCRs the render instead).
    assert any(p.native_text.strip() for p in pages)


@pytest.mark.skipif(
    os.environ.get("RUN_PARSE_E2E") != "1",
    reason="parse e2e is opt-in: set RUN_PARSE_E2E=1 (needs VL server + LLM-provider access)",
)
def test_ocr_reads_what_the_text_layer_cannot(tmp_path):
    """Page 10 is a crisp BART receipt with ZERO native text — every word in
    the output had to come from OCR of the pixels. Page 9 is rotated
    sideways with a mojibake layer — it must yield content, not a crash."""
    import pypdfium2 as pdfium

    from ingestlib.operations.parse import parse

    src = pdfium.PdfDocument(str(_SCAN))
    excerpt = pdfium.PdfDocument.new()
    excerpt.import_pages(src, pages=[8, 9])  # 0-based: pages 9 and 10
    out = tmp_path / "scan-excerpt.pdf"
    excerpt.save(str(out))

    r = parse(out)
    assert r.page_count == 2

    bart = r.pages[1]
    assert bart.native_text.strip() == "", "page 10 must have no text layer"
    md = bart.markdown.upper()
    assert "BART" in md, f"OCR missed the receipt issuer: {bart.markdown[:200]!r}"
    assert "20.00" in md, "OCR missed the printed amount"

    rotated = r.pages[0]
    assert rotated.markdown.strip(), "the rotated page must yield content"
