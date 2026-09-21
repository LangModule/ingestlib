"""Fixtures for the sources suite — a real on-disk SQLite DB and a SourceSpec
factory, plus a scratch-config dir for registry tests.

SQLite is serverless, so the deterministic tests here run ungated in `make test`:
they exercise the real SQL engine/execution path and stub only the LLM calls
(SQL generation, verified-match embeddings), which are the sole remote pieces.
"""
import os

import pytest

pytest.importorskip("sqlalchemy")  # the `sql` extra — present in the dev env

from ingestlib.config import SourceSpec


def _make_rx_db(path) -> None:
    """A tiny prescriptions table: 3 rows, 2 of them 'ready'."""
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE rx (rx_id INTEGER PRIMARY KEY, status TEXT, ready_at TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO rx (rx_id, status, ready_at) VALUES "
            "(1, 'ready', '2026-01-01'), (2, 'ready', '2026-01-02'), (3, 'pending', NULL)"
        ))
    engine.dispose()


@pytest.fixture()
def rx_dsn(tmp_path):
    """A real SQLite file with the `rx` table. Engines are cached per DSN
    process-wide, so reset around the test to keep the pool clean."""
    from ingestlib.sources.sql.engine import reset_engines

    db = tmp_path / "rx.db"
    _make_rx_db(db)
    reset_engines()
    yield f"sqlite:///{db}"
    reset_engines()


@pytest.fixture()
def rx_spec(rx_dsn):
    """Factory → a SourceSpec pointing at the rx SQLite DB. Pass overrides to
    tweak allow / row_limit / verified / etc."""
    def _make(**overrides) -> SourceSpec:
        fields = dict(
            name="rx",
            type="sqlite",
            dsn=rx_dsn,
            description="prescription fills",
            tables={"rx": "one row per prescription; status is ready|pending"},
        )
        fields.update(overrides)
        return SourceSpec(**fields)

    return _make


@pytest.fixture()
def scratch_config(tmp_path):
    """A scratch config dir active via INGESTLIB_CONFIG, isolated from the real
    .env, with config + registry + engine caches reset around the test (mirrors
    tests/test_config.py's fixture, but also drops the source caches)."""
    import ingestlib.config as config_module
    from ingestlib.config import reset_config

    env_before = dict(os.environ)
    dotenv_before = set(config_module._dotenv_keys)
    config_before = config_module._config

    for key in dotenv_before:  # forget what the real .env injected
        os.environ.pop(key, None)
    config_module._dotenv_keys.clear()
    config_module._config = None
    os.environ["INGESTLIB_CONFIG"] = str(tmp_path / "config.yaml")
    reset_config()  # drop registry/engine caches built from the real config
    try:
        yield tmp_path
    finally:
        config_module._config = config_before
        config_module._dotenv_keys.clear()
        config_module._dotenv_keys.update(dotenv_before)
        os.environ.clear()
        os.environ.update(env_before)
        reset_config()
