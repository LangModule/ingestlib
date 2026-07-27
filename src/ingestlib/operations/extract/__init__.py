"""Extract operation — your schema in, cited field values out.

    from pydantic import BaseModel
    from ingestlib.operations import parse, extract

    class Receipt(BaseModel):
        merchant: str
        total: float
        currency: str

    result = parse("expenses.pdf")                     # scans need OCR first
    report = extract(result, schema=Receipt, mode="many")

    for item in report.items:                          # one per receipt found
        print(item.value.merchant, item.value.total, item.citation)
        item.fields["total"].region_ids                # exact blocks on the page
        item.fields["total"].grounded                  # value verified in source

mode="one" extracts a single instance per document (an invoice, a contract);
mode="many" extracts every instance. Citations are verified against the
parse, and every value is grounded against its cited text — a field whose
evidence doesn't check out cannot report high confidence.
"""
from ingestlib.operations.extract.extractor import aextract, extract
from ingestlib.operations.extract.models import (
    ExtractedItem,
    ExtractResult,
    FieldValue,
)

__all__ = ["extract", "aextract", "ExtractResult", "ExtractedItem", "FieldValue"]
