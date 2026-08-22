"""Per-database specifics behind the shared SQLAlchemy Core interface.

Every backend is one SQLAlchemy dialect, so a dialect here is thin: the
`sources.yaml` `type` maps to how a per-query timeout is applied (the SQLAlchemy
URL scheme itself comes from the DSN in .env, so nothing here parses it). Five
thin specs don't warrant a package — one module is cleaner.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Dialect:
    """timeout_sql, if set, is run on the connection before the query — formatted
    with `sec` (seconds) and `ms` (milliseconds). Empty means no server-side
    timeout is available (row_limit still bounds the result)."""
    name: str
    timeout_sql: str = ""


_DIALECTS: dict[str, Dialect] = {
    "postgres": Dialect("postgres", "SET statement_timeout = {ms}"),
    "mysql": Dialect("mysql", "SET SESSION max_execution_time = {ms}"),
    "sqlite": Dialect("sqlite"),   # no server-side statement timeout
    "duckdb": Dialect("duckdb"),   # embedded; row_limit is the bound
    "snowflake": Dialect("snowflake", "ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {sec}"),
}


def get_dialect(type_: str) -> Dialect:
    """The Dialect for a sources.yaml `type`, or a clear error listing the valid ones."""
    dialect = _DIALECTS.get(type_)
    if dialect is None:
        raise ValueError(
            f"unknown SQL source type {type_!r} — one of {sorted(_DIALECTS)}"
        )
    return dialect
