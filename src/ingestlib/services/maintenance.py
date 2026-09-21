"""Event-driven registry backup — triggered on ingest when a threshold trips.

config.yaml's registry.backup {enabled, after_docs, after_days} governs it: after
each successful ingest, if enough new documents or days have passed since the last
backup, pg_dump the registry into the blob store. The state (last backup time +
doc count) is a small JSON marker in the blob store, so there is no per-ingest
write unless a backup actually runs.
"""
import json
from datetime import datetime, timezone
from typing import Any

from ingestlib.config import get_config
from ingestlib.storage import registry
from ingestlib.storage.blobs import get_blob_store
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_STATE_KEY = "registry-backup-state.json"


def _read_state(store: Any) -> dict[str, Any] | None:
    try:
        return json.loads(store.get(_STATE_KEY))
    except Exception:
        return None  # no marker yet, or unreadable → treat as no baseline


def _is_due(cfg: Any, state: dict[str, Any] | None, current_docs: int, now: datetime) -> bool:
    """Whether a backup should run now, given the config thresholds and last state."""
    if state is None:
        return True  # no baseline yet — take the first backup and start the clock
    if cfg.after_docs > 0 and current_docs - int(state.get("doc_count", 0)) >= cfg.after_docs:
        return True
    if cfg.after_days > 0:
        try:
            elapsed = now - datetime.fromisoformat(state["ts"])
        except Exception:
            return True  # unparseable timestamp — back up and rewrite the marker
        if elapsed.days >= cfg.after_days:
            return True
    return False


def maybe_backup_registry() -> None:
    """Back up the registry if config.yaml's registry.backup threshold has tripped.

    Best-effort: any failure is logged, never raised — an ingest must not fail
    because a scheduled backup could not run.
    """
    cfg = get_config().registry.backup
    if not cfg.enabled or (cfg.after_docs <= 0 and cfg.after_days <= 0):
        return
    try:
        store = get_blob_store()
        state = _read_state(store)
        current_docs = registry.count_documents()
        now = datetime.now(timezone.utc)
        if not _is_due(cfg, state, current_docs, now):
            return
        from ingestlib.cli.registry import backup_registry

        key, size = backup_registry()
        store.put(
            _STATE_KEY,
            json.dumps({"ts": now.isoformat(), "doc_count": current_docs}).encode(),
            "application/json",
        )
        logger.info("auto-backup: registry → %s (%d bytes)", key, size)
    except Exception as exc:
        logger.warning("auto-backup skipped (%s: %s)", type(exc).__name__, exc)
