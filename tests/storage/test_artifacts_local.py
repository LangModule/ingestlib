"""Artifact store round-trips on the local backend — always run.

No gate: the local filesystem IS the real backend, so this suite covers the
full artifacts surface (the same surface test_artifacts_e2e exercises
against S3) with zero credentials. Uses a synthetic ParseResult in a
tmp_path-rooted store.
"""
import dataclasses
import json
from pathlib import Path

import pytest

import ingestlib.config as config_module
from ingestlib.config import ArtifactsConfig, get_config
from ingestlib.storage import artifacts
from ingestlib.storage.blobs import LocalBlobStore, get_blob_store, reset_blob_store

_DOC_ID = "local-test-" + "0" * 54


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


@pytest.fixture()
def local_store(tmp_path, monkeypatch):
    """Config switched to a tmp_path-rooted local artifact store."""
    cfg = dataclasses.replace(
        get_config(),
        artifact_store="local",
        artifacts=ArtifactsConfig(path=tmp_path),
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    reset_blob_store()
    yield tmp_path
    reset_blob_store()


def test_selected_backend_is_local(local_store):
    assert isinstance(get_blob_store(), LocalBlobStore)


def test_unknown_backend_raises(local_store, monkeypatch):
    cfg = dataclasses.replace(get_config(), artifact_store="gcs")
    monkeypatch.setattr(config_module, "_config", cfg)
    reset_blob_store()
    with pytest.raises(ValueError, match="local.*s3"):
        get_blob_store()


def test_save_parse_writes_browsable_files(local_store):
    doc_id = artifacts.save_parse(_synthetic_parse_result())
    assert doc_id == _DOC_ID
    root = local_store / "documents" / _DOC_ID
    assert (root / "parse" / "result.json").is_file()
    assert (root / "parse" / "document.md").is_file()
    assert (root / "parse" / "pages" / "page_0001.png").read_bytes() == b"\x89PNG-page"
    assert (root / "parse" / "figures" / "page1_region0_chart.png").read_bytes() == b"\x89PNG-fig"
    assert (root / "meta.json").is_file()
    assert not list(root.rglob("*.tmp")), "atomic writes must leave no temp files"


def test_document_exists_and_load_parse_round_trip(local_store):
    assert artifacts.document_exists(_DOC_ID) is False
    artifacts.save_parse(_synthetic_parse_result())
    assert artifacts.document_exists(_DOC_ID) is True

    loaded = artifacts.load_parse(_DOC_ID)
    page = loaded.pages[0]
    assert page.markdown == "# hello"
    assert page.regions[0].bbox.as_tuple() == (10.0, 20.0, 110.0, 70.0)
    assert page.image_bytes is None  # structure-only by default
    with_images = artifacts.load_parse(_DOC_ID, include_images=True)
    assert with_images.pages[0].image_bytes == b"\x89PNG-page"
    assert with_images.pages[0].figures[0].image_bytes == b"\x89PNG-fig"


def test_classify_split_manifest_round_trip_and_registry(local_store):
    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.operations.split.models import Chunk, Section, SplitResult

    artifacts.save_parse(_synthetic_parse_result())
    artifacts.save_classify(_DOC_ID, ClassifyResult(category="survey", confidence=0.9))
    assert artifacts.load_classify(_DOC_ID).category == "survey"

    chunk = Chunk(chunk_id=0, section="s", text="t", markdown="m",
                  embedding_text="[s]\n\nm", pages=[1], region_ids={1: [0]})
    artifacts.save_split(_DOC_ID, SplitResult(
        sections=[Section(name="s", pages=[1], chunks=[chunk])], pages_used=1,
    ))
    assert artifacts.load_split(_DOC_ID).chunks[0].region_ids == {1: [0]}

    assert artifacts.ingest_complete(_DOC_ID) is False
    artifacts.save_ingest_manifest(_DOC_ID, {"store": "SqliteStore", "dimension": 8})
    assert artifacts.ingest_complete(_DOC_ID) is True
    assert artifacts.load_ingest_manifest(_DOC_ID)["dimension"] == 8

    metas = artifacts.list_documents()
    assert [m.doc_id for m in metas] == [_DOC_ID]
    assert metas[0].category == "survey"
    assert metas[0].chunks == 1


def test_meta_self_heals_from_parse_artifact(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    (local_store / "documents" / _DOC_ID / "meta.json").unlink()
    meta = artifacts.get_document_meta(_DOC_ID)
    assert meta.filename == "synthetic.pdf"
    assert meta.page_count == 1
    assert (local_store / "documents" / _DOC_ID / "meta.json").is_file(), (
        "healing must persist the rebuilt meta"
    )


def test_read_blob_serves_page_images(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    key = artifacts.page_image_key(_DOC_ID, 1)
    assert artifacts.read_blob(key) == b"\x89PNG-page"


def test_delete_document_removes_everything(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    deleted = artifacts.delete_document(_DOC_ID)
    assert deleted >= 5  # result.json + document.md + page + figure + meta
    assert artifacts.document_exists(_DOC_ID) is False
    assert not (local_store / "documents" / _DOC_ID).exists()
    assert artifacts.delete_document(_DOC_ID) == 0


def test_corrupt_meta_is_rebuilt_not_crashed(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    meta_path = local_store / "documents" / _DOC_ID / "meta.json"
    meta_path.write_text("{not json")
    meta = artifacts.get_document_meta(_DOC_ID)
    assert meta.page_count == 1
    assert json.loads(meta_path.read_text())["filename"] == "synthetic.pdf"
