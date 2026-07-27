"""ingest() lifecycle: auto-replace, explicit replaces=, and move detection.

Always run, no gates: the model boundaries are stubbed (lifecycle conftest's
`pipeline` fixture) but artifacts and vectors are REAL — local files +
sqlite — so replacement is proven against the actual registry, manifest,
and deletion paths.
"""

import pytest

from ingestlib.services import aingest
from ingestlib.storage import artifacts
from ingestlib.utils.files import sha256_of_file

from tests.services.conftest import vec


async def test_changed_content_same_path_replaces(pipeline):
    """The core lifecycle promise: editing a file and re-ingesting leaves
    exactly ONE version — the new one — in both stores."""
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"version one")
    first = await aingest(doc, store=pipeline.store)
    assert first.status == "ingested"

    doc.write_bytes(b"version two")
    second = await aingest(doc, store=pipeline.store)

    assert second.status == "replaced"
    assert second.replaced_doc_id == first.doc_id
    assert second.doc_id != first.doc_id
    assert "replace" in second.durations
    # old version fully gone, new fully live
    assert artifacts.document_exists(first.doc_id) is False
    assert artifacts.ingest_complete(second.doc_id) is True
    hits = pipeline.store.query(vec(1.0), top_k=5)
    assert {h.document_id for h in hits} == {second.doc_id}


async def test_same_content_same_path_skips(pipeline):
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"stable content")
    await aingest(doc, store=pipeline.store)
    again = await aingest(doc, store=pipeline.store)
    assert again.status == "skipped"


async def test_same_content_new_path_is_a_move(pipeline):
    """THE skip-path fix: the registry must follow a renamed file, or a later
    sync(old_dir, prune=True) would delete a live document."""
    old = pipeline.corpus / "old-name.pdf"
    old.write_bytes(b"same bytes")
    first = await aingest(old, store=pipeline.store)

    new = pipeline.corpus / "sub" / "new-name.pdf"
    new.parent.mkdir()
    old.rename(new)
    moved = await aingest(new, store=pipeline.store)

    assert moved.status == "moved"
    assert moved.doc_id == first.doc_id
    assert artifacts.find_by_path(new).doc_id == first.doc_id
    assert artifacts.find_by_path(old) is None


async def test_explicit_replaces_supersedes_a_different_path(pipeline):
    """replaces= is the override for 'the new version lives elsewhere'."""
    old = pipeline.corpus / "report-v1.pdf"
    old.write_bytes(b"version one")
    first = await aingest(old, store=pipeline.store)

    new = pipeline.corpus / "report-v2.pdf"
    new.write_bytes(b"version two")
    second = await aingest(new, store=pipeline.store, replaces=first.doc_id)

    assert second.status == "replaced"
    assert second.replaced_doc_id == first.doc_id
    assert artifacts.document_exists(first.doc_id) is False


async def test_unknown_replaces_fails_before_the_pipeline(pipeline):
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"content")
    with pytest.raises(ValueError, match="matches no stored document"):
        await aingest(doc, store=pipeline.store, replaces="f" * 64)
    assert artifacts.list_documents() == [], "nothing may run on a bad replaces"


async def test_replace_scoped_by_namespace(pipeline):
    """A path match in ANOTHER namespace is not a replacement — logical
    identity is (namespace, path)."""
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"version one")
    first = await aingest(doc, store=pipeline.store)

    doc.write_bytes(b"version two")
    other = await aingest(doc, store=pipeline.store, namespace="tenant-a")

    assert other.status == "ingested", "different namespace — no replace"
    assert artifacts.document_exists(first.doc_id) is True


async def test_crash_during_old_deletion_leaves_both_versions(pipeline, monkeypatch):
    """Over-complete, never under-complete: if deleting the old version dies,
    the NEW version is already fully live and the old still exists — sync()
    repairs the duplicate later. Nothing is ever lost."""
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"version one")
    first = await aingest(doc, store=pipeline.store)

    def boom(document_id, namespace=""):
        raise RuntimeError("store down mid-delete")

    monkeypatch.setattr(pipeline.store, "delete_document", boom)
    doc.write_bytes(b"version two")
    with pytest.raises(RuntimeError, match="mid-delete"):
        await aingest(doc, store=pipeline.store)

    new_doc_id = sha256_of_file(doc)
    assert artifacts.ingest_complete(new_doc_id) is True, "new version is live"
    assert artifacts.document_exists(first.doc_id) is True, "old not half-deleted"


async def test_on_stage_reports_the_replace_stage(pipeline):
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"version one")
    await aingest(doc, store=pipeline.store)

    events: list[tuple[str, str]] = []
    doc.write_bytes(b"version two")
    await aingest(doc, store=pipeline.store,
                  on_stage=lambda stage, event: events.append((stage, event)))
    assert ("replace", "start") in events and ("replace", "done") in events
