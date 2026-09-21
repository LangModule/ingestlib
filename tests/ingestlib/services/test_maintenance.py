"""Event-driven registry backup (#5) — the on-ingest trigger and its threshold.

The pg_dump core (backup_registry) is stubbed; these test the decision to run and
the state marker, not pg_dump itself (that lives in tests/registry/test_backup_e2e).
"""
import dataclasses
from datetime import datetime, timedelta, timezone

import ingestlib.config as config_module


def test_is_due_thresholds():
    from ingestlib.config import RegistryBackupConfig
    from ingestlib.services.maintenance import _is_due

    now = datetime(2026, 1, 10, tzinfo=timezone.utc)

    docs = RegistryBackupConfig(enabled=True, after_docs=5, after_days=0)
    assert _is_due(docs, None, 0, now) is True                                   # no baseline
    assert _is_due(docs, {"doc_count": 10, "ts": now.isoformat()}, 12, now) is False  # +2 < 5
    assert _is_due(docs, {"doc_count": 10, "ts": now.isoformat()}, 15, now) is True   # +5 >= 5

    days = RegistryBackupConfig(enabled=True, after_docs=0, after_days=7)
    stale = (now - timedelta(days=8)).isoformat()
    assert _is_due(days, {"doc_count": 0, "ts": stale}, 0, now) is True
    fresh = (now - timedelta(days=3)).isoformat()
    assert _is_due(days, {"doc_count": 0, "ts": fresh}, 0, now) is False


def test_maybe_backup_triggers_and_writes_marker(stack, monkeypatch):
    import json

    import ingestlib.cli.registry as reg_cli
    from ingestlib.config import RegistryBackupConfig, get_config
    from ingestlib.services.maintenance import _STATE_KEY, maybe_backup_registry
    from ingestlib.storage import registry
    from ingestlib.storage.blobs import get_blob_store

    cfg = get_config()
    reg = dataclasses.replace(
        cfg.registry, backup=RegistryBackupConfig(enabled=True, after_docs=1, after_days=0)
    )
    monkeypatch.setattr(config_module, "_config", dataclasses.replace(cfg, registry=reg))

    registry.set_status("doc-x", "ingested")  # count_documents() >= 1

    called = {}

    def fake_backup():
        called["ran"] = True
        return "registry-backups/ts/registry.dump", 42

    monkeypatch.setattr(reg_cli, "backup_registry", fake_backup)

    maybe_backup_registry()

    assert called.get("ran") is True
    state = json.loads(get_blob_store().get(_STATE_KEY))
    assert state["doc_count"] >= 1 and "ts" in state


def test_maybe_backup_disabled_is_a_noop(stack, monkeypatch):
    import ingestlib.cli.registry as reg_cli
    from ingestlib.config import RegistryBackupConfig, get_config
    from ingestlib.services.maintenance import maybe_backup_registry

    cfg = get_config()
    reg = dataclasses.replace(
        cfg.registry, backup=RegistryBackupConfig(enabled=False, after_docs=1)
    )
    monkeypatch.setattr(config_module, "_config", dataclasses.replace(cfg, registry=reg))

    ran = {"called": False}

    def fake_backup():
        ran["called"] = True
        return "k", 1

    monkeypatch.setattr(reg_cli, "backup_registry", fake_backup)
    maybe_backup_registry()
    assert ran["called"] is False
