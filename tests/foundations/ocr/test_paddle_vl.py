"""PaddleOCR-VL smoke tests against the real MLX inference server.

Opt-in via RUN_OCR_E2E=1 — requires mlx_vlm.server running on the configured port
(~10s per page on Apple Silicon).
"""
import os
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_DATA_DIR = _TESTS_DIR / "data"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OCR_E2E") != "1",
    reason="OCR e2e is opt-in: set RUN_OCR_E2E=1 (needs the VL inference server running)",
)


@pytest.fixture(scope="session")
def layout():
    """One real full-pipeline run, shared across every test in this module."""
    from ingestlib.foundations.ocr import paddle_vl
    from ingestlib.operations.parse.loaders import load_pdf

    pages, _ = load_pdf(_DATA_DIR / "pdf" / "clinical-study.pdf", render=True, dpi=200)
    return paddle_vl.run_full_pipeline(pages[0].image_bytes)


def test_regions_detected(layout):
    assert len(layout.regions) > 5


def test_region_ids_are_reading_order(layout):
    assert [r.region_id for r in layout.regions] == list(range(len(layout.regions)))


def test_page_dimensions_populated(layout):
    assert layout.page_width > 0 and layout.page_height > 0


def test_text_and_markdown_assembled(layout):
    assert layout.text.strip() and layout.markdown.strip()


def test_bboxes_within_page(layout):
    for r in layout.regions:
        x1, y1, x2, y2 = r.bbox.normalized(layout.page_width, layout.page_height)
        assert 0 <= x1 <= x2 <= 1.01 and 0 <= y1 <= y2 <= 1.01
