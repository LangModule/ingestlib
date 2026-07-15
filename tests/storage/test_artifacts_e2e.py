"""Artifact store round-trips against real S3. Opt-in via RUN_S3_E2E=1.

Uses a synthetic ParseResult (no OCR/VL server needed) with a sentinel doc_id,
and deletes everything it created.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_S3_E2E") != "1",
    reason="S3 e2e is opt-in: set RUN_S3_E2E=1 (needs AWS credentials; creates real objects)",
)

_DOC_ID = "e2e-test-" + "0" * 56  # sentinel checksum-shaped id, safe to delete


def _synthetic_parse_result():
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
        source_checksum=_DOC_ID,
    )


@pytest.fixture(scope="module")
def saved_doc():
    from ingestlib.storage import artifacts

    doc_id = artifacts.save_parse(_synthetic_parse_result())
    yield doc_id
    artifacts.delete_document(doc_id)


def test_save_returns_checksum_as_doc_id(saved_doc):
    assert saved_doc == _DOC_ID


def test_document_exists_after_save(saved_doc):
    from ingestlib.storage import artifacts

    assert artifacts.document_exists(saved_doc) is True
    assert artifacts.document_exists("f" * 64) is False


def test_load_parse_structure_round_trip(saved_doc):
    from ingestlib.storage import artifacts

    loaded = artifacts.load_parse(saved_doc)
    page = loaded.pages[0]
    assert page.markdown == "# hello"
    assert page.regions[0].region_id == 0
    assert page.regions[0].bbox.as_tuple() == (10.0, 20.0, 110.0, 70.0)
    assert page.image_bytes is None  # structure-only by default
    assert page.figures[0].caption == "Fig 1"


def test_load_parse_with_images_restores_bytes(saved_doc):
    from ingestlib.storage import artifacts

    loaded = artifacts.load_parse(saved_doc, include_images=True)
    assert loaded.pages[0].image_bytes == b"\x89PNG-page"
    assert loaded.pages[0].figures[0].image_bytes == b"\x89PNG-fig"


def test_ingest_complete_requires_the_manifest(saved_doc):
    """Parse artifacts alone must NOT count as fully ingested — only the
    manifest (the pipeline's last write) does, so failed runs get retried."""
    from ingestlib.storage import artifacts

    assert artifacts.document_exists(saved_doc) is True
    assert artifacts.ingest_complete(saved_doc) is False
    artifacts.save_ingest_manifest(saved_doc, {"store": "TestStore", "vector_ids": []})
    assert artifacts.ingest_complete(saved_doc) is True
    assert artifacts.load_ingest_manifest(saved_doc)["store"] == "TestStore"


def test_classify_and_split_round_trip(saved_doc):
    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.operations.split.models import Chunk, Section, SplitResult
    from ingestlib.storage import artifacts

    artifacts.save_classify(saved_doc, ClassifyResult(category="survey", confidence=0.9))
    assert artifacts.load_classify(saved_doc).category == "survey"

    chunk = Chunk(
        chunk_id=0, section="s", text="t", markdown="m",
        embedding_text="[s]\n\nm", pages=[1], region_ids={1: [0]},
    )
    artifacts.save_split(saved_doc, SplitResult(
        sections=[Section(name="s", pages=[1], chunks=[chunk])], pages_used=1,
    ))
    loaded = artifacts.load_split(saved_doc)
    assert loaded.chunks[0].region_ids == {1: [0]}


def test_delete_document_removes_everything():
    from ingestlib.storage import artifacts

    tmp_id = "e2e-delete-" + "1" * 53
    pr = _synthetic_parse_result().model_copy(update={"source_checksum": tmp_id})
    artifacts.save_parse(pr)
    assert artifacts.document_exists(tmp_id)
    deleted = artifacts.delete_document(tmp_id)
    assert deleted >= 3  # result.json + document.md + page + figure
    assert artifacts.document_exists(tmp_id) is False
