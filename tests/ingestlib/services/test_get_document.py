"""get_document() — read one stored document back (registry structure + lazy blobs).

Uses the `pipeline` fixture (stubbed model boundaries, real registry + sqlite +
local blobs), so it exercises the real ingest → get_document round trip.
"""


def test_get_document_round_trip(pipeline):
    from ingestlib.services import get_document, ingest
    from ingestlib.utils.files import sha256_of_file

    source = pipeline.corpus / "report.pdf"
    source.write_bytes(b"%PDF get_document fixture")
    doc_id = sha256_of_file(source)
    ingest(source, store=pipeline.store, namespace="ns")

    doc = get_document(doc_id)
    assert doc is not None
    assert doc.doc_id == doc_id
    assert doc.filename == "report.pdf"
    assert doc.namespace == "ns"
    assert doc.category == "report"
    assert doc.status == "ingested"
    assert doc.page_count == 1 and doc.chunk_count == 1
    assert len(doc.pages) == 1
    assert len(doc.chunks) == 1
    assert doc.chunks[0]["region_ids"] == {1: [0]}

    # lazy blob access
    assert doc.markdown() == "# hello"          # from parse/document.md
    assert doc.page_image(1) == b"\x89PNG-page"  # from parse/pages/*.png

    assert get_document("f" * 64) is None


def test_get_document_excludes_tombstoned_replacement(pipeline):
    from ingestlib.services import get_document, ingest

    source = pipeline.corpus / "report.pdf"
    source.write_bytes(b"v1")
    first = ingest(source, store=pipeline.store)
    source.write_bytes(b"v2")
    second = ingest(source, store=pipeline.store)
    assert second.status == "replaced"

    assert get_document(first.doc_id) is None       # tombstone reads as gone
    assert get_document(second.doc_id) is not None
