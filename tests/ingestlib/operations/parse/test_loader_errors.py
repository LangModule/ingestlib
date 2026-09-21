"""Loader failure translation — every error a user can hit names its fix.

All failures here are REAL: garbage bytes through the real pdfium, a real
missing binary, a real un-decodable image, a real encrypted PDF
(tests/data/pdf/password-protected.pdf — insurance-acord.pdf sealed once
with pikepdf; the loader takes no password, so any password provokes it)."""
from pathlib import Path

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


def test_password_protected_pdf_names_the_fix():
    tests_dir = Path(__file__).resolve().parent
    while tests_dir.name != "tests":
        tests_dir = tests_dir.parent
    sealed = (tests_dir / "data" / "pdf" / "password-protected.pdf").read_bytes()
    with pytest.raises(RuntimeError, match="password-protected") as exc:
        load_pdf_from_bytes(sealed)
    assert "remove the password" in str(exc.value)
