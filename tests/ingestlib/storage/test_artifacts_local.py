"""Artifact store round-trips on the local backend — always run.

No gate: the local filesystem IS the real backend, so this suite covers the
artifact store's BYTES surface (source, page renders, figure crops, document.md
— the same S3 test_artifacts_e2e exercises) with zero credentials. Structure and
metadata live in the registry, covered by tests/registry/ + the lifecycle suites.
Uses a synthetic ParseResult in a tmp_path-rooted store.
"""
import dataclasses
from pathlib import Path

import pytest

import ingestlib.config as config_module
from ingestlib.config import ArtifactsConfig, get_config
from ingestlib.storage import artifacts
from ingestlib.storage.blobs import LocalBlobStore, get_blob_store, reset_blob_store

_DOC_ID = "local-test-" + "0" * 54


def _synthetic_parse_result(doc_id: str = _DOC_ID, source: str = "synthetic.pdf"):
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
        source_path=Path(source),
        source_format="pdf",
        source_checksum=doc_id,
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


def test_save_parse_writes_browsable_bytes(local_store):
    """Bytes only: source (absent here), page renders, figure crops, document.md —
    no result.json/meta.json (those moved to the registry)."""
    doc_id = artifacts.save_parse(_synthetic_parse_result())
    assert doc_id == _DOC_ID
    root = local_store / "documents" / _DOC_ID
    assert (root / "parse" / "document.md").is_file()
    assert (root / "parse" / "pages" / "page_0001.png").read_bytes() == b"\x89PNG-page"
    assert (root / "parse" / "figures" / "page1_region0_chart.png").read_bytes() == b"\x89PNG-fig"
    assert not (root / "parse" / "result.json").exists(), "structure lives in the registry"
    assert not (root / "meta.json").exists(), "metadata lives in the registry"
    assert not list(root.rglob("*.tmp")), "atomic writes must leave no temp files"


def test_read_blob_serves_page_images(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    key = artifacts.page_image_key(_DOC_ID, 1)
    assert artifacts.read_blob(key) == b"\x89PNG-page"


def test_document_markdown_round_trips(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    assert artifacts.document_markdown(_DOC_ID) == "# hello"
    assert artifacts.document_markdown("f" * 64) is None


def test_missing_blobs_flags_absent_essentials(local_store):
    """The blob-store audit half of verify: document.md present, source absent
    (synthetic.pdf never existed on disk, so no source blob was written)."""
    artifacts.save_parse(_synthetic_parse_result())
    missing = artifacts.missing_blobs(_DOC_ID, "synthetic.pdf")
    assert missing == ["source"]
    assert artifacts.missing_blobs("f" * 64, "x.pdf") == ["document.md", "source"]


def test_delete_document_removes_everything(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    deleted = artifacts.delete_document(_DOC_ID)
    assert deleted >= 3  # document.md + page render + figure crop
    assert not (local_store / "documents" / _DOC_ID).exists()
    assert artifacts.delete_document(_DOC_ID) == 0


# The corpus-registry surface (document_exists · ingest_complete · list_documents ·
# get_document_meta · find_by_path · set_source_path · load_split) reads from the
# Postgres registry; it is covered by tests/registry/ + the services/CLI lifecycle suites.
