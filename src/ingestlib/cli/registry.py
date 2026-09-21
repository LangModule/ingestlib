"""`ingestlib registry` — create/upgrade and inspect the internal registry DB.

Drives Alembic from a programmatically built Config pointing at the migrations
shipped inside the ingestlib_registry package. alembic.ini is a dev-only file and
is not in the wheel, so the runtime never reads it.
"""
from pathlib import Path
from urllib.parse import urlsplit


def _alembic_config():
    from alembic.config import Config
    import ingestlib_registry

    migrations = Path(ingestlib_registry.__file__).resolve().parent / "migrations"
    cfg = Config()
    cfg.set_main_option("script_location", str(migrations))
    return cfg


def _target() -> str:
    """host:port/db from the registry URL — never the user or password."""
    from ingestlib_registry.db import registry_url

    parts = urlsplit(registry_url())
    return f"{parts.hostname or '?'}:{parts.port or 5432}/{parts.path.lstrip('/') or '?'}"


def _current_revision() -> str | None:
    from alembic.runtime.migration import MigrationContext
    from ingestlib_registry.db import get_engine

    with get_engine().connect() as conn:
        return MigrationContext.configure(conn).get_current_revision()


def _registry_tables() -> list[str]:
    from sqlalchemy import inspect
    from ingestlib_registry.db import get_engine

    names = inspect(get_engine()).get_table_names()
    return sorted(n for n in names if n != "alembic_version")


def run_registry_init() -> int:
    from alembic import command
    from ingestlib_registry.db import ping

    if not ping():
        raise RuntimeError(
            f"registry unreachable at {_target()} — start it and check INGESTLIB_REGISTRY_URL"
        )
    command.upgrade(_alembic_config(), "head")
    tables = _registry_tables()
    print(f"✓ registry ready at {_target()} — revision {_current_revision()}, {len(tables)} tables")
    print(f"  {', '.join(tables)}")
    return 0


def run_registry_status() -> int:
    from alembic.script import ScriptDirectory
    from ingestlib_registry.db import ping

    if not ping():
        print(f"✗ registry unreachable at {_target()}")
        return 1
    current = _current_revision()
    if current is None:
        print(f"registry reachable at {_target()} but not initialized — run: ingestlib registry init")
        return 1
    head = ScriptDirectory.from_config(_alembic_config()).get_current_head()
    state = "up to date" if current == head else f"behind head {head} — run: ingestlib registry init"
    print(f"✓ registry at {_target()} — revision {current}, {state}")
    print(f"  tables: {', '.join(_registry_tables())}")
    return 0


_BACKUP_PREFIX = "registry-backups"


def _pg_url() -> str:
    """The registry URL in the plain postgresql:// form pg_dump/pg_restore accept."""
    from ingestlib_registry.db import registry_url

    return registry_url().replace("postgresql+psycopg://", "postgresql://", 1)


def _missing_pg_tool(tool: str) -> RuntimeError:
    return RuntimeError(
        f"{tool} not found — install the PostgreSQL client tools "
        f"(macOS: brew install libpq; Linux: apt install postgresql-client)"
    )


def backup_registry() -> tuple[str, int]:
    """pg_dump the registry into the blob store; return (stored key, byte size).

    The reusable core behind `ingestlib registry backup` and the on-ingest
    auto-backup. Raises RuntimeError on an unreachable registry or a pg_dump
    failure; no printing — callers report as they see fit.
    """
    import subprocess
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    from ingestlib.storage.blobs import get_blob_store
    from ingestlib_registry.db import ping

    if not ping():
        raise RuntimeError(
            f"registry unreachable at {_target()} — start it and check INGESTLIB_REGISTRY_URL"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.NamedTemporaryFile(suffix=".dump") as f:
        try:
            subprocess.run(
                ["pg_dump", _pg_url(), "-Fc", "-f", f.name], check=True, capture_output=True
            )
        except FileNotFoundError:
            raise _missing_pg_tool("pg_dump") from None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"pg_dump failed: {exc.stderr.decode(errors='replace')[-500:]}") from exc
        data = _Path(f.name).read_bytes()
    key = f"{_BACKUP_PREFIX}/{stamp}/registry.dump"
    get_blob_store().put(key, data, "application/octet-stream")
    return key, len(data)


def run_registry_backup() -> int:
    """pg_dump the registry and store the dump in the blob store (S3 or local)."""
    key, size = backup_registry()
    print(f"✓ registry backup → {key} ({size:,} bytes)")
    return 0


def restore_registry() -> str | None:
    """Restore the registry from the latest blob-store backup; return the restored
    key, or None when no backup exists. DESTRUCTIVE — pg_restore --clean drops and
    rewrites every table. The reusable core behind the CLI and MCP; no printing."""
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    from ingestlib.storage.blobs import get_blob_store

    store = get_blob_store()
    stamps = store.list_top_dirs(_BACKUP_PREFIX)
    if not stamps:
        return None
    key = f"{_BACKUP_PREFIX}/{max(stamps)}/registry.dump"
    data = store.get(key)
    with tempfile.NamedTemporaryFile(suffix=".dump") as f:
        _Path(f.name).write_bytes(data)
        try:
            subprocess.run(
                ["pg_restore", "--clean", "--if-exists", "--no-owner", "-d", _pg_url(), f.name],
                check=True, capture_output=True,
            )
        except FileNotFoundError:
            raise _missing_pg_tool("pg_restore") from None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"pg_restore failed: {exc.stderr.decode(errors='replace')[-500:]}") from exc
    return key


def run_registry_restore() -> int:
    """Restore the registry from the latest blob-store backup (pg_restore --clean)."""
    key = restore_registry()
    if key is None:
        print(f"✗ no registry backups under {_BACKUP_PREFIX}/ — run `ingestlib registry backup` first")
        return 1
    print(f"✓ registry restored from {key}")
    return 0
