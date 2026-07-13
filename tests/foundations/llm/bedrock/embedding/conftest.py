"""Fixtures for Nova 2 embedding tests: image paths + session-cached embedder."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from ingestlib.foundations.llm.bedrock.embedding import embed_image, embed_text

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_IMAGES_DIR = _TESTS_DIR / "data" / "images"

_EXT_TO_FORMAT = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".gif": "gif",
}


@pytest.fixture(scope="session")
def photo_path() -> Path:
    return _IMAGES_DIR / "photo.jpg"


@pytest.fixture(scope="session")
def doc_text_path() -> Path:
    return _IMAGES_DIR / "document_text.png"


@pytest.fixture(scope="session")
def doc_chart_path() -> Path:
    return _IMAGES_DIR / "document_chart.png"


class _Embedder:
    """Session-cached embedder — same args across tests hit Bedrock once."""

    def __init__(self):
        @lru_cache(maxsize=None)
        def _text(text: str, purpose: str, dim: int) -> tuple:
            return tuple(embed_text(text, purpose=purpose, dimension=dim))

        @lru_cache(maxsize=None)
        def _image(data: bytes, format: str, purpose: str, dim: int, detail_level: str) -> tuple:
            return tuple(
                embed_image(
                    data, format=format, purpose=purpose,
                    dimension=dim, detail_level=detail_level,
                )
            )

        self._text = _text
        self._image = _image

    @staticmethod
    def _read_image(path) -> tuple[bytes, str]:
        path = Path(path)
        return path.read_bytes(), _EXT_TO_FORMAT[path.suffix.lower()]

    def text(self, text: str, purpose: str = "GENERIC_INDEX", dim: int = 1024) -> np.ndarray:
        return np.asarray(self._text(text, purpose, dim), dtype=float)

    def image(
        self,
        path,
        purpose: str = "GENERIC_INDEX",
        dim: int = 1024,
        detail_level: str = "STANDARD_IMAGE",
    ) -> np.ndarray:
        data, fmt = self._read_image(path)
        return np.asarray(self._image(data, fmt, purpose, dim, detail_level), dtype=float)


@pytest.fixture(scope="session")
def embed() -> _Embedder:
    return _Embedder()
