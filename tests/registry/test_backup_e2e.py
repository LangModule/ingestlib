"""registry backup/restore round-trip (pg_dump ↔ blob store).

Opt-in via RUN_REGISTRY_E2E=1 AND the PostgreSQL client tools (pg_dump/pg_restore)
on PATH — skipped otherwise. Backs the registry up into a local blob store, wipes
a row, restores, and checks it came back.
"""
import dataclasses
import os
import shutil

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_REGISTRY_E2E") != "1",
        reason="registry e2e is opt-in: set RUN_REGISTRY_E2E=1",
    ),
    pytest.mark.skipif(
        shutil.which("pg_dump") is None or shutil.which("pg_restore") is None,
        reason="needs the PostgreSQL client tools (pg_dump/pg_restore) on PATH",
    ),
]


def test_backup_then_restore_round_trip(tmp_path, monkeypatch):
    import ingestlib.config as config_module
    from ingestlib_registry import db

    from ingestlib.cli.registry import (
        run_registry_backup,
        run_registry_init,
        run_registry_restore,
    )
    from ingestlib.config import ArtifactsConfig, get_config
    from ingestlib.storage import registry
    from ingestlib.storage.blobs import reset_blob_store

    cfg = dataclasses.replace(
        get_config(), artifact_store="local", artifacts=ArtifactsConfig(path=tmp_path),
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    reset_blob_store()
    db.reset_engine()
    run_registry_init()
    registry.set_status("backup-e2e", "ingested")

    try:
        assert run_registry_backup() == 0
        registry.delete_document("backup-e2e")
        assert registry.document_exists("backup-e2e") is False

        assert run_registry_restore() == 0
        db.reset_engine()
        assert registry.document_exists("backup-e2e") is True
    finally:
        registry.delete_document("backup-e2e")
        reset_blob_store()
        db.reset_engine()
