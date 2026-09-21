"""Durability ledger — verify() audits the registry's chunk_count against what
the vector store actually holds (count_vectors), catching silent vector loss.

Uses the `pipeline` fixture (real SqliteStore + real registry, stubbed embeddings).
"""


def _ingest_two(pipeline):
    from ingestlib.services import ingest
    from ingestlib.utils.files import sha256_of_file

    a = pipeline.corpus / "a.pdf"
    a.write_bytes(b"alpha doc")
    b = pipeline.corpus / "b.pdf"
    b.write_bytes(b"beta doc")
    ingest(a, store=pipeline.store)
    ingest(b, store=pipeline.store)
    return sha256_of_file(a), sha256_of_file(b)


def test_count_vectors_matches_what_was_upserted(pipeline):
    a_id, _ = _ingest_two(pipeline)
    # the pipeline split yields one chunk per doc
    assert pipeline.store.count_vectors(a_id) == 1
    assert pipeline.store.count_vectors("no-such-doc") == 0


def test_verify_reports_durable_then_catches_drift(pipeline):
    from ingestlib.services import verify

    a_id, b_id = _ingest_two(pipeline)

    result = verify(store=pipeline.store)
    assert result.checked == 2 and result.ok and not result.drifted

    # simulate silent vector loss: delete one doc's vectors OUT OF BAND, leaving
    # the registry's chunk_count untouched
    pipeline.store.delete_document(a_id)

    drifted = verify(store=pipeline.store)
    assert not drifted.ok
    ids = {i.doc_id for i in drifted.drifted}
    assert a_id in ids and b_id not in ids
    item = next(i for i in drifted.drifted if i.doc_id == a_id)
    assert item.expected == 1 and item.actual == 0


def test_verify_single_document(pipeline):
    from ingestlib.services import verify

    a_id, _ = _ingest_two(pipeline)
    result = verify(a_id, store=pipeline.store)
    assert result.checked == 1 and result.items[0].doc_id == a_id and result.ok


def test_verify_catches_missing_source_blob(pipeline):
    from ingestlib.services import verify
    from ingestlib.storage import artifacts
    from ingestlib.storage.blobs import get_blob_store

    a_id, b_id = _ingest_two(pipeline)
    # lose the source bytes out of band (a partial/rotted blob store)
    get_blob_store().delete_prefix(artifacts._key(a_id, "source"))

    result = verify(store=pipeline.store)
    assert not result.ok
    item = next(i for i in result.drifted if i.doc_id == a_id)
    assert item.vectors_ok            # the vectors are fine
    assert "source" in item.missing_blobs  # the blob store is not


def test_repair_re_embeds_drifted_vectors(pipeline):
    from ingestlib.services import verify

    a_id, _ = _ingest_two(pipeline)
    pipeline.store.delete_document(a_id)  # silent vector loss
    assert not verify(store=pipeline.store).ok

    repaired = verify(store=pipeline.store, repair=True)
    assert repaired.ok  # re-embedded from the registry chunks
    item = next(i for i in repaired.items if i.doc_id == a_id)
    assert item.repaired and item.actual == item.expected == 1
    # and it sticks on a fresh audit
    assert verify(store=pipeline.store).ok
