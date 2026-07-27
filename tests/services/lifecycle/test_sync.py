"""sync() — the reconciler. Always run, no gates: stubbed model boundaries
(lifecycle `pipeline` fixture), REAL registry + sqlite vectors, so every
phase — plan, execute, repair, prune — is proven against actual state.
"""
from pathlib import Path

import pytest

import ingestlib.services.lifecycle.syncer as syncer_mod
from ingestlib.services import aingest, async_sync, sync
from ingestlib.storage import artifacts

from tests.services.conftest import vec


def _by_action(result):
    out = {}
    for a in result.actions:
        out.setdefault(a.action, []).append(a)
    return out


async def test_full_matrix_in_one_run(pipeline):
    """New, changed, moved, and unchanged files — one sync, four verdicts."""
    corpus = pipeline.corpus
    unchanged = corpus / "unchanged.pdf"
    unchanged.write_bytes(b"stable")
    changed = corpus / "changed.pdf"
    changed.write_bytes(b"v1")
    old_name = corpus / "old-name.pdf"
    old_name.write_bytes(b"moving bytes")
    for f in (unchanged, changed, old_name):
        await aingest(f, store=pipeline.store)

    changed.write_bytes(b"v2")                      # changed
    old_name.rename(corpus / "new-name.pdf")        # moved
    (corpus / "brand-new.pdf").write_bytes(b"new")  # new

    result = await async_sync(corpus, store=pipeline.store)
    got = _by_action(result)

    assert [a.path for a in got["ingest"]] == [str(corpus / "brand-new.pdf")]
    assert [a.path for a in got["replace"]] == [str(corpus / "changed.pdf")]
    assert [a.path for a in got["move"]] == [str(corpus / "new-name.pdf")]
    assert [a.path for a in got["skip"]] == [str(corpus / "unchanged.pdf")]
    assert result.counts == {"ingest": 1, "replace": 1, "move": 1, "skip": 1}
    # replaced old version is gone; exactly 4 documents remain
    assert len(artifacts.list_documents()) == 4


async def test_gone_file_survives_without_prune(pipeline):
    doc = pipeline.corpus / "deleted-later.pdf"
    doc.write_bytes(b"content")
    first = await aingest(doc, store=pipeline.store)
    doc.unlink()

    result = await async_sync(pipeline.corpus, store=pipeline.store)
    assert "prune" not in result.counts
    assert artifacts.document_exists(first.doc_id) is True


async def test_prune_removes_gone_files_but_is_root_scoped(pipeline):
    """Prune deletes the corpus doc whose file is gone — and never touches a
    document recorded OUTSIDE the synced root."""
    staying = pipeline.corpus / "staying.pdf"
    staying.write_bytes(b"still here")  # the scan must find SOMETHING, or
    await aingest(staying, store=pipeline.store)  # the empty-scan guard fires

    inside = pipeline.corpus / "gone.pdf"
    inside.write_bytes(b"inside content")
    gone = await aingest(inside, store=pipeline.store)

    elsewhere_dir = pipeline.root / "elsewhere"
    elsewhere_dir.mkdir()
    outside = elsewhere_dir / "other.pdf"
    outside.write_bytes(b"outside content")
    kept = await aingest(outside, store=pipeline.store)
    outside.unlink()  # its file is gone too — but it's not under the root
    inside.unlink()

    result = await async_sync(pipeline.corpus, prune=True, store=pipeline.store)

    assert [a.doc_id for a in _by_action(result)["prune"]] == [gone.doc_id]
    assert artifacts.document_exists(gone.doc_id) is False
    assert artifacts.document_exists(kept.doc_id) is True, "root scoping"
    hits = pipeline.store.query(vec(1.0), top_k=5)
    assert gone.doc_id not in {h.document_id for h in hits}, "vectors pruned too"


async def test_empty_scan_with_prune_refuses(pipeline):
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"content")
    await aingest(doc, store=pipeline.store)
    doc.unlink()  # the corpus dir is now empty — prune would delete everything

    with pytest.raises(ValueError, match="NO ingestible files.*refusing"):
        await async_sync(pipeline.corpus, prune=True, store=pipeline.store)
    assert len(artifacts.list_documents()) == 1, "nothing may be deleted"

    # dry_run is the inspection tool — it must show the plan, not refuse
    plan = await async_sync(
        pipeline.corpus, prune=True, dry_run=True, store=pipeline.store
    )
    assert "prune" in plan.counts


async def test_dry_run_plans_everything_and_changes_nothing(pipeline):
    corpus = pipeline.corpus
    changed = corpus / "changed.pdf"
    changed.write_bytes(b"v1")
    await aingest(changed, store=pipeline.store)
    changed.write_bytes(b"v2")
    (corpus / "new.pdf").write_bytes(b"new")
    goner = corpus / "gone.pdf"
    goner.write_bytes(b"bye")
    gone = await aingest(goner, store=pipeline.store)
    goner.unlink()

    before_registry = {m.doc_id for m in artifacts.list_documents()}
    result = await async_sync(corpus, prune=True, dry_run=True, store=pipeline.store)

    assert result.dry_run is True
    got = _by_action(result)
    assert [a.path for a in got["ingest"]] == [str(corpus / "new.pdf")]
    assert [a.path for a in got["replace"]] == [str(corpus / "changed.pdf")]
    assert [a.doc_id for a in got["prune"]] == [gone.doc_id]
    assert {m.doc_id for m in artifacts.list_documents()} == before_registry, (
        "dry_run must not touch the registry"
    )


async def test_repair_deletes_the_older_duplicate(pipeline):
    """Two documents claiming one path — a crashed replace's trace. The
    newest created_at keeps the path; the older is removed."""
    doc = pipeline.corpus / "report.pdf"
    doc.write_bytes(b"v1")
    first = await aingest(doc, store=pipeline.store)
    doc.write_bytes(b"v2")
    second = await aingest(doc, store=pipeline.store)
    assert second.status == "replaced"

    # resurrect the crashed state: v1's artifacts exist again, same path
    from tests.services.conftest import synthetic_parse_result

    artifacts.save_parse(synthetic_parse_result(first.doc_id, doc))
    artifacts._patch_meta(first.doc_id, created_at="2000-01-01T00:00:00+00:00")
    artifacts._patch_meta(second.doc_id, created_at="2026-01-01T00:00:00+00:00")

    result = await async_sync(pipeline.corpus, store=pipeline.store)

    repairs = _by_action(result)["repair"]
    assert [a.doc_id for a in repairs] == [first.doc_id]
    assert artifacts.document_exists(first.doc_id) is False
    assert artifacts.document_exists(second.doc_id) is True


async def test_narrow_glob_never_prunes_files_it_skipped(pipeline):
    """The is_file() guard: a document whose file EXISTS but is excluded by
    the glob must survive a prune."""
    pdf = pipeline.corpus / "keep.pdf"
    pdf.write_bytes(b"pdf content")
    kept = await aingest(pdf, store=pipeline.store)
    (pipeline.corpus / "target.png").write_bytes(b"png content")

    result = await async_sync(
        pipeline.corpus, glob="**/*.png", prune=True, store=pipeline.store
    )

    assert "prune" not in result.counts
    assert artifacts.document_exists(kept.doc_id) is True


async def test_namespace_isolation(pipeline):
    """Syncing the default namespace never touches another tenant's docs."""
    doc = pipeline.corpus / "tenant-doc.pdf"
    doc.write_bytes(b"tenant content")
    tenant = await aingest(doc, store=pipeline.store, namespace="tenant-a")
    doc.unlink()

    result = await async_sync(pipeline.corpus, prune=True, store=pipeline.store)

    assert result.actions == []
    assert artifacts.document_exists(tenant.doc_id) is True


async def test_one_bad_file_does_not_abandon_the_sync(pipeline, monkeypatch):
    corpus = pipeline.corpus
    (corpus / "bad.pdf").write_bytes(b"will fail")
    (corpus / "good.pdf").write_bytes(b"will succeed")

    real_aingest = syncer_mod.aingest

    async def flaky(path, **kwargs):
        if Path(path).name == "bad.pdf":
            raise RuntimeError("parser exploded")
        return await real_aingest(path, **kwargs)

    monkeypatch.setattr(syncer_mod, "aingest", flaky)
    result = await async_sync(corpus, store=pipeline.store)

    got = _by_action(result)
    assert [a.path for a in got["error"]] == [str(corpus / "bad.pdf")]
    assert "parser exploded" in got["error"][0].detail
    assert [a.path for a in got["ingest"]] == [str(corpus / "good.pdf")]
    assert len(result.errors) == 1


def test_sync_wrapper_and_not_a_directory(pipeline):
    with pytest.raises(ValueError, match="not a directory"):
        sync(pipeline.corpus / "missing-dir")
    result = sync(pipeline.corpus, store=pipeline.store)  # empty dir, no prune
    assert result.actions == [] and result.dry_run is False
