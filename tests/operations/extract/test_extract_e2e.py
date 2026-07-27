"""Extract against the real stack. Opt-in via RUN_EXTRACT_E2E=1.

Two proofs:
  - mode="many" over a real SCANNED receipt bundle (a two-page excerpt of
    expenses.pdf through the full OCR parse) must find the BART receipt at
    exactly 20.00, grounded, with region-level provenance
  - mode="one" over a raw NATIVE path (no OCR server needed) must pull
    Apple's total net sales with page-level provenance

The many-mode proof needs the VL server + the LLM provider; the native
proof needs only the LLM provider.
"""
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from ingestlib.operations.extract import extract

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_PDF = _TESTS_DIR / "data" / "pdf"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EXTRACT_E2E") != "1",
    reason="extract e2e is opt-in: set RUN_EXTRACT_E2E=1 (needs the LLM "
           "provider; the scanned case also needs the VL server)",
)


class Receipt(BaseModel):
    merchant: str
    total: float
    currency: str


class Financials(BaseModel):
    company: str
    fiscal_year: int
    total_net_sales: float


def test_many_mode_finds_the_bart_receipt_in_a_real_scan(tmp_path):
    import pypdfium2 as pdfium

    from ingestlib.operations.parse import parse

    src = pdfium.PdfDocument(str(_PDF / "expenses.pdf"))
    excerpt = pdfium.PdfDocument.new()
    excerpt.import_pages(src, pages=[8, 9])  # rotated receipts + the BART page
    out = tmp_path / "receipts.pdf"
    excerpt.save(str(out))

    result = parse(out)
    report = extract(result, schema=Receipt, mode="many")

    assert report.items, "the scan holds receipts"
    bart = [
        item for item in report.items
        if "bart" in item.value.merchant.lower()
    ]
    assert bart, f"BART receipt not found among {[i.value.merchant for i in report.items]}"
    item = bart[0]
    assert item.value.total == pytest.approx(20.00)
    total = item.fields["total"]
    assert total.grounded is True, "20.00 must be read from the cited OCR text"
    assert total.region_ids, "region-level provenance on the parse path"
    assert item.pages == [2], "the BART receipt is the excerpt's second page"


def test_one_mode_native_path_extracts_grounded_financials():
    """No OCR server involved — the raw path reads the native text layer."""
    report = extract(_PDF / "finance-10k.pdf", schema=Financials, mode="one")

    assert len(report.items) == 1
    item = report.items[0]
    assert "apple" in item.value.company.lower()
    assert item.value.total_net_sales == pytest.approx(383285.0)

    sales = item.fields["total_net_sales"]
    assert sales.grounded is True
    assert sales.region_ids == {}, "native path is page-level only"
    assert sales.pages, "page citation present"
