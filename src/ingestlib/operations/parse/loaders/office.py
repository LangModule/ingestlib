"""Office document loader (DOCX, PPTX).

Converts the source through LibreOffice into PDF bytes, then delegates to the
PDF loader — both loading shapes: rendered pages (load_office, the parse
pipeline's input) and lightweight content (load_office_content: native text +
embedded images for classify/split). Downstream output is produced from the
intermediate PDF exactly as it would be for a native PDF input.
"""
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from ingestlib.operations.parse.loaders.pdf import (
    ContentPage,
    LoadedPage,
    load_pdf_content_from_bytes,
    load_pdf_from_bytes,
)


# Extensions this loader accepts. Not a superset of DOCX/PPTX by design —
# other office formats (ODT, ODP, etc.) would need explicit support here.
OfficeExtension = Literal["docx", "pptx"]

# LibreOffice's macOS Homebrew cask exposes the binary as `soffice`. On Linux
# installs the same name works via `libreoffice-core`.
_LIBREOFFICE_BIN = "soffice"

# Big decks with many images can take real time to convert; be generous.
_CONVERSION_TIMEOUT_SECONDS = 120


def _convert_to_pdf_bytes(office_bytes: bytes, ext: OfficeExtension) -> bytes:
    """Run LibreOffice headless to convert the source bytes into PDF bytes.

    The TemporaryDirectory is used only as scratch space for the LibreOffice
    subprocess. It is deleted when this function returns and the PDF bytes are
    already in memory by then.
    """
    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source = tmp / f"input.{ext}"
        source.write_bytes(office_bytes)

        # Unique user-profile per run keeps LibreOffice from clashing with a
        # GUI instance the user may have open. The profile lives under the
        # same TemporaryDirectory and is deleted with it.
        profile = tmp / "profile"

        try:
            subprocess.run(
                [
                    _LIBREOFFICE_BIN,
                    "--headless",
                    "--nologo",
                    "--nofirststartwizard",
                    f"-env:UserInstallation=file://{profile}",
                    "--convert-to", "pdf",
                    "--outdir", str(tmp),
                    str(source),
                ],
                check=True,
                timeout=_CONVERSION_TIMEOUT_SECONDS,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            # capture_output swallows stderr — surface it or failures
            # (missing fonts, corrupt files) are undiagnosable
            stderr = (exc.stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(
                f"LibreOffice failed to convert the {ext} input"
                + (f":\n{stderr[-2000:]}" if stderr else " (no stderr output)")
            ) from exc

        pdf_files = list(tmp.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(
                f"LibreOffice did not produce a PDF for the {ext} input"
            )
        return pdf_files[0].read_bytes()


def load_office_from_bytes(
    office_bytes: bytes,
    ext: OfficeExtension,
    *,
    render: bool = True,
    dpi: int = 200,
) -> tuple[list[LoadedPage], dict[str, Any]]:
    """Load a DOCX or PPTX from bytes.

    Converts through LibreOffice to PDF, then returns whatever the PDF loader
    produces — same (pages, metadata) shape as load_pdf_from_bytes().
    """
    pdf_bytes = _convert_to_pdf_bytes(office_bytes, ext)
    return load_pdf_from_bytes(pdf_bytes, render=render, dpi=dpi)


def load_office(
    path: Path,
    *,
    render: bool = True,
    dpi: int = 200,
) -> tuple[list[LoadedPage], dict[str, Any]]:
    """Path convenience wrapper — reads `path` and delegates to load_office_from_bytes."""
    ext = _validated_ext(path)
    return load_office_from_bytes(
        path.read_bytes(), ext=ext, render=render, dpi=dpi
    )


def load_office_content(path: Path) -> tuple[list[ContentPage], dict[str, Any]]:
    """DOCX/PPTX → PDF → native text + embedded images per page (no rendering)."""
    ext = _validated_ext(path)
    pdf_bytes = _convert_to_pdf_bytes(path.read_bytes(), ext)
    return load_pdf_content_from_bytes(pdf_bytes)


def _validated_ext(path: Path) -> OfficeExtension:
    ext = path.suffix.lower().lstrip(".")
    if ext not in ("docx", "pptx"):
        raise ValueError(
            f"Unsupported office extension: {ext!r}. Supported: 'docx', 'pptx'"
        )
    return cast(OfficeExtension, ext)
