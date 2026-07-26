"""Image loader (PNG, JPEG, WebP) — one image becomes a one-page document.

An image has no text layer and no physical page size, so the pipeline treats
it exactly like a scanned page: OCR reads the render, and provenance stays in
pixel space (UI overlays use bbox.normalized(), which needs only pixel dims).

The content shape hands the image itself over as the page's one embedded
image — classify works on it through the vision LLM with no OCR server; split
has no text to label pages with and points the caller at parse() instead.
"""
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ingestlib.operations.parse.loaders.pdf import (
    _MAX_IMAGE_SIDE,
    ContentPage,
    LoadedPage,
)


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(
            "not a readable image — the file appears corrupt or is an "
            "unsupported encoding"
        ) from exc
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def _to_png(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_image_from_bytes(image_bytes: bytes) -> tuple[list[LoadedPage], dict[str, Any]]:
    """One image → a single LoadedPage at its native resolution, empty text layer."""
    img = _open_image(image_bytes)
    page = LoadedPage(
        image_bytes=_to_png(img),
        native_text="",
        width=img.width,
        height=img.height,
    )
    return [page], {}


def load_image(path: Path) -> tuple[list[LoadedPage], dict[str, Any]]:
    """Path convenience wrapper — reads `path` and delegates to load_image_from_bytes."""
    return load_image_from_bytes(path.read_bytes())


def load_image_content(path: Path) -> tuple[list[ContentPage], dict[str, Any]]:
    """Content shape: no text, the (LLM-sized) image itself as the page's image."""
    img = _open_image(path.read_bytes())
    if max(img.size) > _MAX_IMAGE_SIDE:
        img.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    return [ContentPage(text="", images=[_to_png(img)])], {}
