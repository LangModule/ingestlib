"""Layer A — the registry Postgres (ingestlib's internal spine).

Verifies the read-write role works, the schema is at head, and — the reason
we're on Postgres at all — the read-only role can read everything but cannot
write. Opt-in via RUN_INFRA_E2E=1; skips if the registry is unreachable.
"""
import os

import pytest

from tests.infra._util import infra_e2e, pg_url_plain, ro_url_from

pytestmark = infra_e2e

psycopg = pytest.importorskip("psycopg")

_RW = pg_url_plain(
    os.environ.get("INGESTLIB_REGISTRY_URL", "postgresql://ingestlib:pw@localhost:5433/ingestlib")
)
_RO = ro_url_from(_RW)

_TABLES = ["documents", "pages", "regions", "sections", "chunks", "extractions", "collections"]


def _connect(url: str):
    try:
        conn = psycopg.connect(url, autocommit=True)
    except Exception as exc:  # unreachable / role absent
        pytest.skip(f"registry not reachable at {url.rsplit('@', 1)[-1]}: {exc}")
    return conn


def test_rw_connection_and_select():
    with _connect(_RW) as conn:
        assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_schema_is_at_head_with_all_tables():
    with _connect(_RW) as conn:
        present = {
            r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            )
        }
        for table in _TABLES:
            assert table in present, f"missing table {table!r}"
        assert "alembic_version" in present
        assert conn.execute("SELECT count(*) FROM alembic_version").fetchone()[0] == 1


def test_rw_can_write_and_delete():
    with _connect(_RW) as conn:
        try:
            conn.execute("INSERT INTO documents (doc_id) VALUES ('infra-rw-probe')")
            assert conn.execute(
                "SELECT count(*) FROM documents WHERE doc_id='infra-rw-probe'"
            ).fetchone()[0] == 1
        finally:
            conn.execute("DELETE FROM documents WHERE doc_id='infra-rw-probe'")


def test_ro_role_reads_every_table():
    with _connect(_RO) as conn:
        for table in _TABLES:
            conn.execute(f"SELECT count(*) FROM {table}")  # no error == readable


def test_ro_role_cannot_write():
    with _connect(_RO) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("INSERT INTO documents (doc_id) VALUES ('infra-ro-probe')")
