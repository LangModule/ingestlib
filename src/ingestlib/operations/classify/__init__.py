"""Classify operation — document-type classification, independent of parse.

    from ingestlib.operations.classify import classify

    classify("invoice.pdf")                          # standalone — no OCR
    classify(parse_result)                           # pipeline — reuses the parse
    classify(parse_result, categories={...})         # constrained to your labels (≤20)
    classify(doc, target_pages="1,3,5-7", max_pages=5)   # read only these pages

Rules and page settings can also live in rules.yaml (beside config.yaml) —
used whenever no explicit arguments are given (categories={} forces
open-ended). ≤20 pages classify in one call — text plus up to 4 document
images (embedded pictures standalone, figure crops from a ParseResult).
Larger documents map-reduce over 20-page chunks with a 100-page hard cap.
"""
from ingestlib.operations.classify.classifier import aclassify, classify
from ingestlib.operations.classify.models import CategoryScore, ClassifyResult

__all__ = ["classify", "aclassify", "ClassifyResult", "CategoryScore"]
