"""Format-specific loaders that normalize each source type into per-page records.

Two loading shapes per format:
    load_<fmt> / load_<fmt>_from_bytes          — rendered pages + native text
                                                  → (list[LoadedPage], metadata)
    load_<fmt>_content[_from_bytes]             — native text + embedded images,
                                                  no rasterization
                                                  → (list[ContentPage], metadata)

The rendered shape feeds the parse pipeline; the content shape is the cheap
path for operations that read documents without layout (classify, split).
"""
from ingestlib.operations.parse.loaders.office import (
    OfficeExtension,
    load_office,
    load_office_content,
    load_office_from_bytes,
)
from ingestlib.operations.parse.loaders.pdf import (
    ContentPage,
    LoadedPage,
    load_pdf,
    load_pdf_content,
    load_pdf_content_from_bytes,
    load_pdf_from_bytes,
)

__all__ = [
    "ContentPage",
    "LoadedPage",
    "OfficeExtension",
    "load_office",
    "load_office_content",
    "load_office_from_bytes",
    "load_pdf",
    "load_pdf_content",
    "load_pdf_content_from_bytes",
    "load_pdf_from_bytes",
]
