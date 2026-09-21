"""Image inputs — a single image becomes a one-page document. Pure, always run."""
from pathlib import Path

from PIL import Image as PILImage

from ingestlib.operations.parse.detector import detect_format
from ingestlib.operations.parse.loaders import load_image, load_image_content

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_PHOTO = _TESTS_DIR / "data" / "images" / "photo.jpg"


def test_image_extensions_detect():
    assert detect_format(Path("scan.png")) == "png"
    assert detect_format(Path("scan.JPG")) == "jpeg"
    assert detect_format(Path("scan.jpeg")) == "jpeg"
    assert detect_format(Path("scan.webp")) == "webp"


def test_load_image_is_one_page_with_png_render():
    pages, metadata = load_image(_PHOTO)
    assert len(pages) == 1
    page = pages[0]
    assert page.image_bytes.startswith(b"\x89PNG")
    assert page.native_text == ""
    assert page.width > 0 and page.height > 0
    assert metadata == {}


def test_load_image_content_is_the_image_itself():
    pages, _ = load_image_content(_PHOTO)
    assert len(pages) == 1
    assert pages[0].text == ""
    assert len(pages[0].images) == 1
    img = PILImage.open(__import__("io").BytesIO(pages[0].images[0]))
    assert max(img.size) <= 1600, "LLM-bound images are downscaled"


def test_webp_loads_like_any_other_image():
    """WebP is a claimed input format — prove it through the real loader,
    not just the extension table."""
    scan = _TESTS_DIR / "data" / "images" / "document_scan.webp"
    pages, _ = load_image(scan)
    assert len(pages) == 1
    assert pages[0].image_bytes.startswith(b"\x89PNG"), "renders normalize to PNG"

    content_pages, _ = load_image_content(scan)
    assert len(content_pages[0].images) == 1
