"""Registry CLI + schema e2e against the live `registry` compose service.

Opt-in via RUN_REGISTRY_E2E=1 with the container up (INGESTLIB_REGISTRY_URL
default: postgresql://ingestlib:pw@localhost:5433/ingestlib). Exercises the real
migrations, CASCADE, the read-only role, and `registry init`/`status` end to end —
no stubs. The read-only checks assume the compose init.sql ran (role ingestlib_ro);
override its DSN with INGESTLIB_REGISTRY_RO_URL for a BYO-Postgres deployment.
"""
import os
from urllib.parse import urlsplit

import pytest

pytest.importorskip("sqlalchemy")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REGISTRY_E2E") != "1",
    reason="registry e2e is opt-in: set RUN_REGISTRY_E2E=1 (needs the compose `registry` service up)",
)

_EXPECTED = {
    "documents", "pages", "regions", "sections", "chunks", "extractions", "collections",
}


def _db():
    from ingestlib_registry import db

    return db


def _table_names() -> set[str]:
    from sqlalchemy import inspect

    return set(inspect(_db().get_engine()).get_table_names()) - {"alembic_version"}


def _drop_all() -> None:
    from sqlalchemy import text

    order = "collections, extractions, chunks, sections, regions, pages, documents, alembic_version"
    with _db().get_engine().begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {order} CASCADE"))
    _db().reset_engine()


def _head() -> str:
    from alembic.script import ScriptDirectory

    from ingestlib.cli.registry import _alembic_config

    return ScriptDirectory.from_config(_alembic_config()).get_current_head()


def _ro_url() -> str:
    override = os.environ.get("INGESTLIB_REGISTRY_RO_URL")
    if override:
        return override
    parts = urlsplit(_db().registry_url())
    return f"postgresql+psycopg://ingestlib_ro:ro_pw@{parts.hostname}:{parts.port}/{parts.path.lstrip('/')}"


@pytest.fixture(autouse=True)
def _clean_engine():
    _db().reset_engine()
    yield
    _db().reset_engine()


@pytest.fixture
def initialized():
    from ingestlib.cli.registry import run_registry_init

    assert run_registry_init() == 0


def test_init_creates_all_tables_at_head():
    from ingestlib.cli.registry import _current_revision, run_registry_init

    _drop_all()
    assert run_registry_init() == 0
    assert _table_names() == _EXPECTED
    assert _current_revision() == _head()


def test_init_is_idempotent(initialized):
    from ingestlib.cli.registry import run_registry_init

    assert run_registry_init() == 0
    assert run_registry_init() == 0
    assert _table_names() == _EXPECTED


def test_status_up_to_date_after_init(initialized, capsys):
    from ingestlib.cli.registry import run_registry_status

    assert run_registry_status() == 0
    assert "up to date" in capsys.readouterr().out


def test_status_on_empty_db_is_not_initialized(capsys):
    from ingestlib.cli.registry import run_registry_init, run_registry_status

    _drop_all()
    assert run_registry_status() == 1
    assert "not initialized" in capsys.readouterr().out
    assert run_registry_init() == 0  # restore so order can't strand the DB


def test_cascade_delete_removes_children(initialized):
    from sqlalchemy import func, select

    from ingestlib_registry.models import (
        Chunk, Document, Extraction, Page, Region, Section,
    )

    db = _db()
    doc_id = "e2e-cascade"
    children = (Page, Region, Section, Chunk, Extraction)

    with db.session_scope() as s:
        s.add(Document(doc_id=doc_id))
        s.flush()  # parent first — no ORM relationships, so the UOW won't reorder for us
        s.add(Page(doc_id=doc_id, page_num=1))
        s.add(Region(doc_id=doc_id, page_num=1, region_id=0, bbox={}))
        s.add(Section(doc_id=doc_id, ord=0))
        s.add(Chunk(doc_id=doc_id, chunk_id=0))
        s.add(Extraction(doc_id=doc_id, schema_name="invoice"))

    with db.session_scope() as s:
        for model in children:
            n = s.execute(
                select(func.count()).select_from(model).where(model.doc_id == doc_id)
            ).scalar()
            assert n == 1, f"{model.__tablename__} should have 1 child before delete"

    with db.session_scope() as s:
        s.delete(s.get(Document, doc_id))

    with db.session_scope() as s:
        for model in children:
            n = s.execute(
                select(func.count()).select_from(model).where(model.doc_id == doc_id)
            ).scalar()
            assert n == 0, f"{model.__tablename__} child survived the cascade"


def test_ro_role_can_read_but_not_write(initialized):
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import ProgrammingError

    ro = create_engine(_ro_url(), future=True)
    try:
        with ro.connect() as conn:
            conn.execute(text("SELECT count(*) FROM documents"))  # read is allowed
        with pytest.raises(ProgrammingError):
            with ro.begin() as conn:
                conn.execute(text("INSERT INTO documents (doc_id) VALUES ('ro-nope')"))
    finally:
        ro.dispose()


def test_doctor_check_ok_when_initialized(initialized):
    from ingestlib.cli.doctor import check_registry

    status, detail = check_registry()
    assert status == "ok"
    assert "revision" in detail


def test_doctor_check_warns_when_unreachable(monkeypatch):
    from ingestlib.cli.doctor import check_registry

    monkeypatch.setenv(
        "INGESTLIB_REGISTRY_URL", "postgresql://ingestlib:pw@localhost:5599/ingestlib"
    )
    _db().reset_engine()
    status, detail = check_registry()
    assert status == "warn"
    assert "unreachable" in detail
