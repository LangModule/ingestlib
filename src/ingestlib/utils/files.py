"""File utilities shared across operations and services."""
import hashlib
from pathlib import Path

_CHUNK_BYTES = 64 * 1024


def sha256_of_file(path: Path) -> str:
    """Streaming SHA-256 hex digest of the file at `path` — the document ID."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()
