"""Map a file path to its SourceFormat via extension."""
from pathlib import Path

from ingestlib.operations.parse.models import SourceFormat


_EXT_TO_FORMAT: dict[str, SourceFormat] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(_EXT_TO_FORMAT.keys()))
SUPPORTED_FORMATS: tuple[SourceFormat, ...] = tuple(sorted(set(_EXT_TO_FORMAT.values())))


def detect_format(path: Path) -> SourceFormat:
    """Return the SourceFormat for `path` based on its extension (case-insensitive).

    Raises ValueError with the supported-extension list if the extension is unknown.
    """
    ext = path.suffix.lower()
    if ext not in _EXT_TO_FORMAT:
        raise ValueError(
            f"Unsupported file format: {ext!r}. "
            f"Supported extensions: {list(SUPPORTED_EXTENSIONS)}"
        )
    return _EXT_TO_FORMAT[ext]
