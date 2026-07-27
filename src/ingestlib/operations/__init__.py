"""Document operations — parse, classify, split, extract.

Each operation works standalone and composes into the ingestion pipeline:

    from ingestlib.operations import parse, classify, split, extract

    result = parse("report.pdf")                       # OCR + enrichment
    label  = classify(result)                          # document type
    chunks = split(result, category=label.category)    # RAG-ready chunks
    fields = extract(result, schema=MySchema)          # cited field values

parse is the only operation that runs OCR; the others accept either a
ParseResult (reusing it) or a raw file path (native text + embedded images).
"""
from ingestlib.operations.classify import aclassify, classify
from ingestlib.operations.extract import aextract, extract
from ingestlib.operations.parse import aparse, parse
from ingestlib.operations.split import asplit, split

__all__ = [
    "parse",
    "aparse",
    "classify",
    "aclassify",
    "split",
    "asplit",
    "extract",
    "aextract",
]
