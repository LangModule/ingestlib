"""registry_url() scheme normalization. No server needed; runs in the fast suite.

The engine must always speak psycopg v3, so both driver-less Postgres schemes are
rewritten — `postgresql://` and the older `postgres://` that managed providers
still emit (SQLAlchemy 2.0 dropped the bare `postgres` dialect). A URL that already
names a driver is left alone.
"""
import pytest

from ingestlib_registry.db import registry_url


@pytest.mark.parametrize(
    "given,expected",
    [
        ("postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgres://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        # already names a driver → untouched
        ("postgresql+psycopg://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
    ],
)
def test_scheme_is_normalized_to_psycopg(monkeypatch, given, expected):
    monkeypatch.setenv("INGESTLIB_REGISTRY_URL", given)
    assert registry_url() == expected


def test_default_url_when_env_unset(monkeypatch):
    monkeypatch.delenv("INGESTLIB_REGISTRY_URL", raising=False)
    # the built-in default is a plain postgresql:// → normalized to psycopg
    assert registry_url().startswith("postgresql+psycopg://")


def test_empty_env_falls_back_to_default(monkeypatch):
    # a blank `INGESTLIB_REGISTRY_URL=` line must not yield "" → default instead
    monkeypatch.setenv("INGESTLIB_REGISTRY_URL", "")
    assert registry_url().startswith("postgresql+psycopg://")
