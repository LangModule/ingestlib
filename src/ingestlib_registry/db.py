"""Engine + session for the registry DB, resolved from INGESTLIB_REGISTRY_URL.

Kept free of any `ingestlib` import so the registry stays a standalone package that
ingestlib depends on (one way), never the reverse. Sync psycopg (v3) throughout —
ingestlib has no event loop to share, so an async driver would buy nothing.
"""
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Matches config.py / .env.example; overridden by INGESTLIB_REGISTRY_URL.
_DEFAULT_URL = "postgresql://ingestlib:pw@localhost:5433/ingestlib"

_lock = threading.Lock()
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def registry_url() -> str:
    """The registry connection URL, normalized to the psycopg (v3) driver.

    Both driver-less Postgres schemes map to psycopg v3: `postgresql://` and the
    older `postgres://` that many managed providers (Heroku/Render/some Neon)
    still emit — SQLAlchemy 2.0 dropped the bare `postgres` dialect, so it must
    be rewritten. A URL that already names a driver (`postgresql+psycopg://`,
    etc.) is left untouched.
    """
    url = os.environ.get("INGESTLIB_REGISTRY_URL") or _DEFAULT_URL  # unset OR empty → default
    for scheme in ("postgresql://", "postgres://"):
        if url.startswith(scheme):
            return "postgresql+psycopg://" + url[len(scheme):]
    return url


def _ensure() -> tuple[Engine, sessionmaker[Session]]:
    """The engine + session factory, built once under the lock and returned together."""
    global _engine, _session_factory
    with _lock:
        if _engine is None:
            _engine = create_engine(
                registry_url(),
                future=True,
                pool_pre_ping=True,   # drop dead connections before handing them out
                pool_size=5,
                max_overflow=10,
                pool_recycle=1800,    # recycle after 30 min to dodge server-side idle cuts
                connect_args={"application_name": "ingestlib_registry"},
            )
            _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
        return _engine, _session_factory


def get_engine() -> Engine:
    """The shared engine (lazily created on first call, then cached)."""
    return _ensure()[0]


def get_session() -> Session:
    """A new session bound to the shared engine — caller closes it (or use a `with`).

    Reads the factory returned by the locked _ensure(), so a concurrent
    reset_engine() cannot null it between the check and the call.
    """
    return _ensure()[1]()


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional session: commit on success, roll back on error, always close."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> bool:
    """True if the registry answers `SELECT 1`, False on any connection error."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def reset_engine() -> None:
    """Dispose the engine and drop the cache; next call re-reads INGESTLIB_REGISTRY_URL."""
    global _engine, _session_factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _session_factory = None
