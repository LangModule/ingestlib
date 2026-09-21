"""Artifact store BYTES round-trip against real S3. Opt-in via RUN_S3_E2E=1.

Mirrors test_artifacts_local against the S3 backend: source/page/figure/document.md
bytes written and read back, nothing else — structure + metadata live in the
Postgres registry (covered by tests/registry/ + the lifecycle suites), not S3.
Uses a synthetic ParseResult (no OCR/VL server) with a sentinel doc_id, and
deletes everything it created.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_S3_E2E") != "1",
    reason="S3 e2e is opt-in: set RUN_S3_E2E=1 (needs AWS credentials; creates real objects)",
)

_DOC_ID = "e2e-test-" + "0" * 56  # sentinel checksum-shaped id, safe to delete


def _synthetic_parse_result(doc_id: str = _DOC_ID):
    from ingestlib.foundations.ocr.models import BoundingBox, Region
    from ingestlib.operations.parse.models import FigureImage, PageResult, ParseResult

    region = Region(
        region_type="chart",
        bbox=BoundingBox(x=10, y=20, width=100, height=50),
        region_id=0,
        text="chart data",
        content="| a | b |",
    )
    fig = FigureImage(
        region_id=0, region_type="chart", image_bytes=b"\x89PNG-fig", caption="Fig 1"
    )
    page = PageResult(
        page_num=1,
        text="hello",
        markdown="# hello",
        regions=[region],
        figures=[fig],
        native_text="hello native",
        image_bytes=b"\x89PNG-page",
        page_width=100,
        page_height=200,
    )
    return ParseResult(
        pages=[page],
        source_path=Path("synthetic.pdf"),
        source_format="pdf",
        source_checksum=doc_id,
    )


@pytest.fixture(scope="module")
def saved_doc():
    from ingestlib.storage import artifacts

    doc_id = artifacts.save_parse(_synthetic_parse_result())
    yield doc_id
    artifacts.delete_document(doc_id)


def test_save_returns_checksum_as_doc_id(saved_doc):
    assert saved_doc == _DOC_ID


def test_page_render_round_trips_on_s3(saved_doc):
    from ingestlib.storage import artifacts

    key = artifacts.page_image_key(saved_doc, 1)
    assert artifacts.read_blob(key) == b"\x89PNG-page"


def test_document_markdown_round_trips_on_s3(saved_doc):
    from ingestlib.storage import artifacts

    assert artifacts.document_markdown(saved_doc) == "# hello"
    assert artifacts.document_markdown("f" * 64) is None


def test_missing_blobs_flags_absent_source(saved_doc):
    from ingestlib.storage import artifacts

    # synthetic.pdf never existed on disk, so no source blob was written
    assert artifacts.missing_blobs(saved_doc, "synthetic.pdf") == ["source"]


def test_delete_document_removes_everything():
    from ingestlib.storage import artifacts

    tmp_id = "e2e-delete-" + "1" * 53
    pr = _synthetic_parse_result(tmp_id)
    artifacts.save_parse(pr)
    deleted = artifacts.delete_document(tmp_id)
    assert deleted >= 3  # document.md + page render + figure crop
    assert artifacts.delete_document(tmp_id) == 0
