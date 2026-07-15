"""PDF loader.

Two loading shapes:
  load_pdf / load_pdf_from_bytes                  — rendered pages + native text
                                                    (the parse pipeline's input)
  load_pdf_content / load_pdf_content_from_bytes  — native text + EMBEDDED raster
                                                    images only, no rasterization
                                                    (classify/split's lightweight input)

Both also return a metadata dictionary of the PDF's own properties.
"""
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image


class LoadedPage(NamedTuple):
    """Per-page loader output. Pipeline converts each entry into a PageResult.

    image_bytes  — rendered PNG at the requested DPI; None when render=False
    native_text  — embedded text-layer content; empty string if the PDF has none
    width        — pixels at DPI when rendered; PDF points (1/72") otherwise
    height       — same
    """
    image_bytes: bytes | None
    native_text: str
    width: int
    height: int


# PDF properties surfaced into ParseResult.source_metadata. Keys not in this
# tuple are ignored to keep the metadata dict lean.
_METADATA_KEYS: tuple[str, ...] = (
    "Title",
    "Author",
    "Subject",
    "Keywords",
    "Creator",
    "Producer",
    "CreationDate",
    "ModDate",
)


def _render_page_png(page: Any, dpi: int) -> bytes:
    """Rasterize one PdfPage to PNG bytes at the requested DPI."""
    bitmap = page.render(scale=dpi / 72.0)
    try:
        buf = BytesIO()
        bitmap.to_pil().save(buf, format="PNG")
        return buf.getvalue()
    finally:
        bitmap.close()


def _page_dims(page: Any, *, render: bool, dpi: int) -> tuple[int, int]:
    """Return (width, height) in pixels-at-DPI when rendering, else in PDF points."""
    w_pt, h_pt = page.get_size()
    if render:
        scale = dpi / 72.0
        return int(w_pt * scale), int(h_pt * scale)
    return int(w_pt), int(h_pt)


def _extract_metadata(pdf: pdfium.PdfDocument) -> dict[str, Any]:
    """Read PDF properties into a plain dict, lowercasing keys and skipping absent ones."""
    out: dict[str, Any] = {}
    for key in _METADATA_KEYS:
        try:
            val = pdf.get_metadata_value(key)
        except Exception:
            continue
        if val:
            out[key.lower()] = val
    return out


def load_pdf_from_bytes(
    pdf_bytes: bytes,
    *,
    render: bool = True,
    dpi: int = 200,
) -> tuple[list[LoadedPage], dict[str, Any]]:
    """Load a PDF from bytes.

    Returns (pages, metadata). When render=False the loader skips rasterization
    and each LoadedPage has image_bytes=None.
    """
    # native handles are closed explicitly — GC finalizers are too lazy for a
    # long-running server parsing large documents
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        pages: list[LoadedPage] = []
        for page in pdf:
            textpage = page.get_textpage()
            try:
                native_text = textpage.get_text_range() or ""
            finally:
                textpage.close()
            image_bytes = _render_page_png(page, dpi) if render else None
            width, height = _page_dims(page, render=render, dpi=dpi)
            page.close()
            pages.append(LoadedPage(
                image_bytes=image_bytes,
                native_text=native_text,
                width=width,
                height=height,
            ))

        return pages, _extract_metadata(pdf)
    finally:
        pdf.close()


def load_pdf(
    path: Path,
    *,
    render: bool = True,
    dpi: int = 200,
) -> tuple[list[LoadedPage], dict[str, Any]]:
    """Path convenience wrapper — reads `path` and delegates to load_pdf_from_bytes."""
    return load_pdf_from_bytes(path.read_bytes(), render=render, dpi=dpi)


# ---------- lightweight content loading (no rasterization) ----------


class ContentPage(NamedTuple):
    """Per-page output of the content loaders: text + embedded raster images.

    images — PNG-encoded, largest first, icon-sized images filtered out,
             oversized ones downscaled. A scanned page surfaces naturally:
             its full-page scan IS an embedded image.
    """
    text: str
    images: list[bytes]


# Embedded images smaller than this on either side are icons/logos — skipped.
_MIN_IMAGE_SIDE = 300
# Larger images (slide backgrounds, full-page scans) downscale to this cap
# before being handed to an LLM.
_MAX_IMAGE_SIDE = 1600
_MAX_IMAGES_PER_PAGE = 3


def _extract_embedded_images(page: Any) -> list[bytes]:
    """Raster images embedded in the page, filtered, largest first, downscaled."""
    pils: list[Image.Image] = []
    for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=4):
        try:
            bitmap = obj.get_bitmap(render=False)
            try:
                pil = bitmap.to_pil()
            finally:
                bitmap.close()
        except Exception:  # malformed/unsupported image object — skip it
            continue
        if pil.width >= _MIN_IMAGE_SIDE and pil.height >= _MIN_IMAGE_SIDE:
            pils.append(pil)

    pils.sort(key=lambda im: im.width * im.height, reverse=True)
    out: list[bytes] = []
    for pil in pils[:_MAX_IMAGES_PER_PAGE]:
        if max(pil.size) > _MAX_IMAGE_SIDE:
            pil.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        buf = BytesIO()
        pil.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def load_pdf_content_from_bytes(
    pdf_bytes: bytes,
) -> tuple[list[ContentPage], dict[str, Any]]:
    """Load native text + embedded images per page — no page rendering.

    The cheap path for operations that read a document without needing layout
    (classify, split): text comes from the text layer, images are the actual
    pictures inside the PDF rather than rasterized pages.
    """
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        pages: list[ContentPage] = []
        for page in pdf:
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_range() or ""
            finally:
                textpage.close()
            pages.append(ContentPage(text=text, images=_extract_embedded_images(page)))
            page.close()
        return pages, _extract_metadata(pdf)
    finally:
        pdf.close()


def load_pdf_content(path: Path) -> tuple[list[ContentPage], dict[str, Any]]:
    """Path convenience wrapper — reads `path` and delegates to load_pdf_content_from_bytes."""
    return load_pdf_content_from_bytes(path.read_bytes())
