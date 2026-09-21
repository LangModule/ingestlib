"""1c-ii — the ingestor dual-writes to the registry through a real aingest.

Opt-in via RUN_REGISTRY_E2E=1 (needs the `registry` compose service up). The
`pipeline` fixture stubs the four model boundaries (parse/classify/split/embed)
but keeps artifact + vector writes real, so this exercises the ACTUAL ingestor
wiring — the registry mirror-write at each stage — end to end.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REGISTRY_E2E") != "1",
    reason="registry e2e is opt-in: set RUN_REGISTRY_E2E=1 (needs the compose `registry` service up)",
)


def test_aingest_mirror_writes_the_registry(pipeline):
    from ingestlib_registry import db

    from ingestlib.cli.registry import run_registry_init
    from ingestlib.services.ingest import ingest
    from ingestlib.storage import registry
    from ingestlib.utils.files import sha256_of_file

    db.reset_engine()
    run_registry_init()

    source = pipeline.corpus / "report.pdf"
    source.write_bytes(b"%PDF-1.4 dual-write fixture bytes")
    doc_id = sha256_of_file(source)
    registry.delete_document(doc_id)

    try:
        result = ingest(source, store=pipeline.store, namespace="ns")
        assert result.status == "ingested"

        got = registry.get_document(doc_id)
        assert got is not None, "aingest should have mirror-written the registry"

        doc = got["document"]
        assert doc["namespace"] == "ns"
        assert doc["filename"] == "report.pdf"
        assert doc["source_format"] == "pdf"
        assert doc["category"] == "report"           # from save_classify
        assert doc["classify_confidence"] == 0.9
        assert doc["status"] == "ingested"            # from save_status
        assert doc["vector_store"] == "SqliteStore"   # from save_embed
        assert doc["vector_count"] == 1
        assert doc["embedded_at"] is not None
        assert doc["page_count"] == 1
        assert doc["chunk_count"] == 1

        assert len(got["pages"]) == 1
        assert len(got["regions"]) == 1
        assert len(got["chunks"]) == 1
        assert got["chunks"][0]["region_ids"] == {1: [0]}
    finally:
        registry.delete_document(doc_id)
        db.reset_engine()
