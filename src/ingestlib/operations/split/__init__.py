"""Split operation — sections + natural chunks for RAG, independent of parse.

    from ingestlib.operations.split import split

    split("report.pdf")                              # standalone — no OCR
    split(parse_result)                              # pipeline — region provenance
    split(parse_result, category="research_paper")   # breadcrumb includes doc type

Three passes: section vocabulary discovery → per-page labels → within-section
natural chunk boundaries. Chunks never split tables/figures, carry a context
breadcrumb in embedding_text, and map back to parse regions for citations.
500-page hard cap.
"""
from ingestlib.operations.split.models import Chunk, Section, SplitResult, VocabEntry
from ingestlib.operations.split.splitter import asplit, split

__all__ = ["split", "asplit", "SplitResult", "Section", "Chunk", "VocabEntry"]
