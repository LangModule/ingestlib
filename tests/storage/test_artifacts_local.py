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
    assert meta.source_path.endswith("synthetic.pdf"), (
        "the rebuild must include the logical identity"
    )
    assert (local_store / "documents" / _DOC_ID / "meta.json").is_file(), (
        "healing must persist the rebuilt meta"
    )


def test_read_blob_serves_page_images(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    key = artifacts.page_image_key(_DOC_ID, 1)
    assert artifacts.read_blob(key) == b"\x89PNG-page"


def test_load_on_unknown_doc_id_names_the_fix(local_store):
    """A doc_id that was never stored must not surface a raw path error —
    the message names the missing artifact, the call that produces it, and
    list_documents()."""
    from pydantic import BaseModel

    class _Schema(BaseModel):
        field: str

    with pytest.raises(FileNotFoundError, match="parse.*list_documents"):
        artifacts.load_parse("f" * 64)
    with pytest.raises(FileNotFoundError, match="classify.*list_documents"):
        artifacts.load_classify("f" * 64)
    with pytest.raises(FileNotFoundError, match="split.*list_documents"):
        artifacts.load_split("f" * 64)
    with pytest.raises(FileNotFoundError, match="ingest.*list_documents"):
        artifacts.load_ingest_manifest("f" * 64)
    with pytest.raises(FileNotFoundError, match="_Schema.*list_documents"):
        artifacts.load_extract("f" * 64, _Schema)


def test_extract_round_trip_revalidates_values(local_store):
    """save_extract/load_extract on the real local backend — values come back
    as instances of the caller's schema, provenance intact."""
    from pydantic import BaseModel

    from ingestlib.operations.extract import ExtractedItem, ExtractResult, FieldValue

    class Receipt(BaseModel):
        merchant: str
        total: float

    result = ExtractResult(
        items=[ExtractedItem(
            value=Receipt(merchant="BART", total=20.0),
            fields={"total": FieldValue(confidence=0.9, region_ids={10: [3]},
                                        pages=[10], grounded=True)},
            pages=[10],
        )],
        schema_name="Receipt",
        mode="many",
        pages_used=16,
    )
    artifacts.save_extract(_DOC_ID, result)

    loaded = artifacts.load_extract(_DOC_ID, Receipt)
    item = loaded.items[0]
    assert isinstance(item.value, Receipt) and item.value.total == 20.0
    assert item.fields["total"].region_ids == {10: [3]}
    assert item.fields["total"].grounded is True
    assert loaded.schema_name == "Receipt" and loaded.pages_used == 16


def test_load_classify_before_classify_is_the_same_clear_error(local_store):
    """Parsed but never classified — the standalone-user path."""
    artifacts.save_parse(_synthetic_parse_result())
    with pytest.raises(FileNotFoundError, match="classify"):
        artifacts.load_classify(_DOC_ID)


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


# ---------- logical identity (lifecycle) ----------


def test_save_parse_records_logical_identity(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    meta = artifacts.get_document_meta(_DOC_ID)
    assert meta.source_path == str(Path("synthetic.pdf").resolve())
    assert meta.namespace == ""  # namespace arrives with the ingest manifest


def test_manifest_patches_namespace_into_meta(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    artifacts.save_ingest_manifest(_DOC_ID, {"namespace": "tenant-a", "dimension": 8})
    assert artifacts.get_document_meta(_DOC_ID).namespace == "tenant-a"


def test_pre_lifecycle_meta_gains_identity_without_losing_stage_fields(local_store):
    """A meta.json written before source_path/namespace existed heals both in
    from the parse artifact and the ingest manifest — and keeps its
    category/chunk counts (patch, not rebuild)."""
    from ingestlib.operations.classify.models import ClassifyResult

    artifacts.save_parse(_synthetic_parse_result())
    artifacts.save_classify(_DOC_ID, ClassifyResult(category="survey", confidence=0.9))
    artifacts.save_ingest_manifest(_DOC_ID, {"namespace": "tenant-a"})

    # simulate the pre-lifecycle file: strip the identity fields
    meta_path = local_store / "documents" / _DOC_ID / "meta.json"
    old = json.loads(meta_path.read_text())
    del old["source_path"], old["namespace"]
    meta_path.write_text(json.dumps(old))

    meta = artifacts.get_document_meta(_DOC_ID)
    assert meta.source_path.endswith("synthetic.pdf")
    assert meta.namespace == "tenant-a"
    assert meta.category == "survey", "healing must not lose stage fields"
    persisted = json.loads(meta_path.read_text())
    assert persisted["source_path"] == meta.source_path, "heal must persist"


def test_find_by_path(local_store):
    artifacts.save_parse(_synthetic_parse_result())

    hit = artifacts.find_by_path("synthetic.pdf")
    assert hit is not None and hit.doc_id == _DOC_ID
    assert artifacts.find_by_path("other.pdf") is None
    assert artifacts.find_by_path("synthetic.pdf", namespace="tenant-a") is None, (
        "namespace scopes the logical identity"
    )


def test_find_by_path_duplicate_resolves_to_newest(local_store):
    """Two documents claiming one path (a crashed replace) — the newest
    created_at wins; sync() repairs the duplicate later."""
    other_id = "local-test-" + "1" * 54
    artifacts.save_parse(_synthetic_parse_result())
    artifacts.save_parse(_synthetic_parse_result(doc_id=other_id))
    # same recorded path for both; force a deterministic created_at order
    artifacts._patch_meta(_DOC_ID, created_at="2026-01-01T00:00:00+00:00")
    artifacts._patch_meta(other_id, created_at="2026-02-01T00:00:00+00:00")

    hit = artifacts.find_by_path("synthetic.pdf")
    assert hit is not None and hit.doc_id == other_id


def test_set_source_path_repoints_identity(local_store):
    artifacts.save_parse(_synthetic_parse_result())
    artifacts.set_source_path(_DOC_ID, "moved/renamed.pdf")
    meta = artifacts.get_document_meta(_DOC_ID)
    assert meta.source_path == str(Path("moved/renamed.pdf").resolve())
    assert artifacts.find_by_path("moved/renamed.pdf").doc_id == _DOC_ID
    assert artifacts.find_by_path("synthetic.pdf") is None
