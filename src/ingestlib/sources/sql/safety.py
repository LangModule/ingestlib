"""Statement guardrails — the read-only boundary a query passes before the driver.

The database ROLE is the hard floor (a read-only connection can't write, whatever
the SQL says). This is defense-in-depth on top of it: only allowed statement
types run, a single statement at a time, and a LIMIT is injected when missing so
a generated query can never scan unbounded. Verified queries pass the same gate.
"""
import re

_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


class UnsafeQuery(ValueError):
    """A generated or verified query violated the source's permission boundary."""


def guard(sql: str, *, allow: tuple[str, ...]) -> str:
    """Reject anything outside `allow`; return the cleaned single statement.

    allow is the source's statement-type allowlist (default ("select",)). A
    query whose first keyword isn't allowed, or that stacks multiple statements,
    raises UnsafeQuery — before it ever reaches the driver.
    """
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("empty query")
    if ";" in cleaned:
        raise UnsafeQuery("multiple statements are not allowed — one query only")
    first = cleaned.split(None, 1)[0].lower()
    if first not in allow:
        raise UnsafeQuery(
            f"statement type {first!r} is not allowed for this source "
            f"(allowed: {', '.join(allow)})"
        )
    return cleaned


def with_limit(sql: str, row_limit: int) -> str:
    """Append a LIMIT when the query has none — a generated query can't run away.

    Best-effort: a query that already limits (any LIMIT clause) is left alone.
    """
    if row_limit <= 0 or _LIMIT_RE.search(sql):
        return sql
    return f"{sql}\nLIMIT {row_limit}"
