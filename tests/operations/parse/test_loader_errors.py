"""Loader failure translation — every error a user can hit names its fix.

All failures here are REAL: garbage bytes through the real pdfium, a real
missing binary, a real un-decodable image. The password-protected branch is
untested — crafting an encrypted PDF needs a dependency we don't want."""
import pytest

from ingestlib.operations.parse.loaders import (
    load_image_from_bytes,
    load_office_from_bytes,
    load_pdf_from_bytes,
)
from ingestlib.operations.parse.loaders import office as office_module
from ingestlib.operations.parse.loaders.pdf import load_pdf_content_from_bytes


def test_corrupt_pdf_names_the_problem():
    with pytest.raises(RuntimeError, match="not a readable PDF"):
        load_pdf_from_bytes(b"this is definitely not a pdf")


def test_corrupt_pdf_content_shape_names_the_problem():
    with pytest.raises(RuntimeError, match="not a readable PDF"):
        load_pdf_content_from_bytes(b"garbage bytes")


def test_missing_libreoffice_names_the_install_command(monkeypatch):
    """A real FileNotFoundError from a nonexistent binary must translate."""
    monkeypatch.setattr(office_module, "_LIBREOFFICE_BIN", "soffice-definitely-missing")
    with pytest.raises(RuntimeError, match="LibreOffice is not installed") as exc:
        load_office_from_bytes(b"anything", ext="docx")
    assert "brew install" in str(exc.value) and "apt install" in str(exc.value)
    assert "PDF inputs work without it" in str(exc.value)


def test_corrupt_image_names_the_problem():
    with pytest.raises(RuntimeError, match="not a readable image"):
        load_image_from_bytes(b"not an image at all")
