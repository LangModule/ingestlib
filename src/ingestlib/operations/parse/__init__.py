"""Parse operation — one mode, any supported format, full-fidelity output.

    from ingestlib.operations.parse import parse
    result = parse("report.pdf")

Pipeline per page: PaddleOCR-VL (layout + text + tables + formulas) → LLM
enrichment (charts → data tables, figures → descriptions + PNG crops) → LLM
review (per-region corrections) → markdown assembly.
"""
from ingestlib.operations.parse.detector import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
    detect_format,
)
from ingestlib.operations.parse.models import (
    FigureImage,
    PageResult,
    ParseResult,
    SourceFormat,
)
from ingestlib.operations.parse.pipeline import aparse, parse

__all__ = [
    "parse",
    "aparse",
    "ParseResult",
    "PageResult",
    "FigureImage",
    "SourceFormat",
    "detect_format",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMATS",
]
