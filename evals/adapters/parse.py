"""Run ingestlib's parse over a document → its whole-document markdown.

Exercises the full parse pipeline (OCR + enrichment + assembly), so it needs the
PaddleOCR-VL inference server on a GPU.
"""
from pathlib import Path

from ingestlib.operations import aparse


async def predict_markdown(doc_path: Path) -> str:
    """A PDF or image → the assembled whole-document markdown."""
    return (await aparse(doc_path)).markdown
