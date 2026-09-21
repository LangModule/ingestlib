"""Format detection — pure logic, always run."""
from pathlib import Path

import pytest

from ingestlib.operations.parse import SUPPORTED_EXTENSIONS, detect_format


@pytest.mark.parametrize("name,expected", [
    ("report.pdf", "pdf"),
    ("REPORT.PDF", "pdf"),
    ("contract.docx", "docx"),
    ("deck.pptx", "pptx"),
    ("photo.png", "png"),
    ("scan.jpg", "jpeg"),
    ("scan.JPEG", "jpeg"),
    ("page.webp", "webp"),
])
def test_supported_extensions_detect(name, expected):
    assert detect_format(Path(name)) == expected


@pytest.mark.parametrize("name", [
    "multi.tiff", "multi.tif", "photo.heic",  # image formats outside the supported trio
    "notes.txt", "legacy.doc", "legacy.ppt", "archive.zip", "noext",
])
def test_unsupported_extensions_raise(name):
    with pytest.raises(ValueError, match="Unsupported file format"):
        detect_format(Path(name))


def test_supported_extensions_constant():
    assert set(SUPPORTED_EXTENSIONS) == {
        ".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".webp",
    }
