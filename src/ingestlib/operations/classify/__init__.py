"""Classify operation — document-type classification, independent of parse.

    from ingestlib.operations.classify import classify

    classify("invoice.pdf")                          # standalone — no OCR
    classify(parse_result)                           # pipeline — reuses the parse
    classify(parse_result, categories={...})         # constrained to your labels

≤20 pages classify in one Nova call — text plus up to 4 document images
(embedded pictures standalone, figure crops from a ParseResult). Larger
documents map-reduce over 20-page chunks with a 100-page hard cap.
"""
from ingestlib.operations.classify.classifier import aclassify, classify
from ingestlib.operations.classify.models import CategoryScore, ClassifyResult

__all__ = ["classify", "aclassify", "ClassifyResult", "CategoryScore"]
