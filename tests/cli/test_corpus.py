"""Corpus CLI (ingest/sync/list/remove/backfill) driven through main().

The stack is real (local artifacts + sqlite under tmp_path); only the model
boundaries are stubbed, so `ingestlib ingest` exercises the true registry,
replacement, and deletion — capsys checks the printed surface and exit codes.
"""
import dataclasses

import pytest

import ingestlib.config as config_module
from ingestlib.config import ArtifactsConfig, SqliteConfig, get_config
from ingestlib.cli import main
from ingestlib.storage import artifacts
from ingestlib.storage.blobs import reset_blob_store


@pytest.fixture()
def cli_stack(tmp_path, monkeypatch):
    """Local artifacts + sqlite; model boundaries stubbed; a corpus dir.

    default_store() is called fresh inside the commands, so it picks up this
    config — no store threading needed through the CLI.
    """
    cfg = dataclasses.replace(
        get_config(),
        artifact_store="local",
        artifacts=ArtifactsConfig(path=tmp_path / "artifacts"),
        vector_store="sqlite",
        sqlite=SqliteConfig(path=tmp_path / "vectors.db"),
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    reset_blob_store()

    import importlib

    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.operations.split.models import Chunk, Section, SplitResult
    from ingestlib.utils.files import sha256_of_file

    ingestor = importlib.import_module("ingestlib.services.ingest.ingestor")

    def _parse_result(path):
        from ingestlib.foundations.ocr.models import BoundingBox, Region
        from ingestlib.operations.parse.models import PageResult, ParseResult

        region = Region(region_type="text",
                        bbox=BoundingBox(x=1, y=1, width=1, height=1),
                        region_id=0, text="hi", content="hi")
        return ParseResult(
            pages=[PageResult(page_num=1, text="hi", markdown="hi",
                              regions=[region], figures=[], native_text="hi",
                              image_bytes=b"\x89PNG", page_width=1, page_height=1)],
            source_path=path, source_format="pdf",
            source_checksum=sha256_of_file(path),
        )

    async def fake_aparse(path, *, dpi=200):
        from pathlib import Path
        return _parse_result(Path(path))

    async def fake_aclassify(source, categories=None, *, target_pages=None, max_pages=None):
        return ClassifyResult(category="report", confidence=0.9)

    async def fake_asplit(source, *, category=None, max_chunk_tokens=768,
                          vocabulary=None, unmatched=None):
        chunk = Chunk(chunk_id=0, section="body", heading="h", text="hi",
                      markdown="hi", embedding_text="[doc › body › h]\n\nhi",
                      pages=[1], region_ids={1: [0]})
        return SplitResult(sections=[Section(name="body", pages=[1], chunks=[chunk])],
                           pages_used=1)

    async def fake_embed(text, purpose="GENERIC_INDEX", dimension=1024):
        return [1.0] + [0.0] * 7

    monkeypatch.setattr(ingestor, "aparse", fake_aparse)
    monkeypatch.setattr(ingestor, "aclassify", fake_aclassify)
    monkeypatch.setattr(ingestor, "asplit", fake_asplit)
    monkeypatch.setattr(ingestor, "aembed_text", fake_embed)
    # backfill re-embeds through the ingestor's _embed_chunks → the same
    # stubbed aembed_text, so no extra seam is needed

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    yield corpus
    reset_blob_store()


def test_ingest_a_file_then_list(cli_stack, capsys):
    doc = cli_stack / "report.pdf"
    doc.write_bytes(b"content one")

    assert main(["ingest", str(doc)]) == 0
    out = capsys.readouterr().out
    assert "ingested: 1 chunk(s)" in out
    assert "1/1 succeeded" in out

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "report.pdf" in out and "1 document(s)" in out


def test_ingest_missing_file_is_friendly(cli_stack, capsys):
    """A typo'd path — the most likely first-use mistake — must not leak a
    raw OS errno (the library's 'errors carry their fix' principle)."""
    assert main(["ingest", str(cli_stack / "nope.pdf")]) == 1
    out = capsys.readouterr().out
    assert "file not found" in out
    assert "Errno" not in out


def test_ingest_a_folder(cli_stack, capsys):
    (cli_stack / "a.pdf").write_bytes(b"aaa")
    (cli_stack / "b.pdf").write_bytes(b"bbb")
    (cli_stack / "notes.txt").write_bytes(b"ignored")  # unsupported ext

    assert main(["ingest", str(cli_stack)]) == 0
    out = capsys.readouterr().out
    assert "2/2 succeeded" in out  # the .txt is not counted


def test_ingest_reports_replacement(cli_stack, capsys):
    doc = cli_stack / "report.pdf"
    doc.write_bytes(b"v1")
    main(["ingest", str(doc)])
    capsys.readouterr()
    doc.write_bytes(b"v2")

    assert main(["ingest", str(doc)]) == 0
    assert "replaced" in capsys.readouterr().out
    assert len(artifacts.list_documents()) == 1


def test_sync_dry_run_changes_nothing(cli_stack, capsys):
    (cli_stack / "new.pdf").write_bytes(b"new")
    assert main(["sync", str(cli_stack), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "would ingest: new.pdf" in out
    assert "plan:" in out
    assert artifacts.list_documents() == [], "dry-run must not ingest"


def test_sync_then_prune(cli_stack, capsys):
    keep = cli_stack / "keep.pdf"
    keep.write_bytes(b"keep")
    goner = cli_stack / "goner.pdf"
    goner.write_bytes(b"bye")
    main(["ingest", str(cli_stack)])
    capsys.readouterr()
    goner.unlink()

    assert main(["sync", str(cli_stack), "--prune"]) == 0
    out = capsys.readouterr().out
    assert "prune: goner.pdf" in out
    assert {m.filename for m in artifacts.list_documents()} == {"keep.pdf"}


def test_remove_by_path(cli_stack, capsys):
    doc = cli_stack / "report.pdf"
    doc.write_bytes(b"content")
    main(["ingest", str(doc)])
    capsys.readouterr()

    assert main(["remove", str(doc)]) == 0
    assert "removed report.pdf" in capsys.readouterr().out
    assert artifacts.list_documents() == []


def test_remove_unknown_target_exits_one(cli_stack, capsys):
    assert main(["remove", "nope.pdf"]) == 1
    assert "no stored document" in capsys.readouterr().out


def test_list_empty_and_namespace_scope(cli_stack, capsys):
    assert main(["list"]) == 0
    assert "no documents stored" in capsys.readouterr().out

    doc = cli_stack / "t.pdf"
    doc.write_bytes(b"tenant")
    main(["ingest", str(doc), "--namespace", "tenant-a"])
    capsys.readouterr()

    assert main(["list", "--namespace", "other"]) == 0
    assert "no documents stored" in capsys.readouterr().out
    assert main(["list", "--namespace", "tenant-a"]) == 0
    assert "t.pdf" in capsys.readouterr().out


def test_backfill_rebuilds(cli_stack, capsys):
    doc = cli_stack / "report.pdf"
    doc.write_bytes(b"content")
    main(["ingest", str(doc)])
    capsys.readouterr()

    assert main(["backfill"]) == 0
    assert "backfilled 1 document(s)" in capsys.readouterr().out
