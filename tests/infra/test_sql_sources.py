"""Layer A — the structured-retrieval SQL source server (mysql).

Verifies the container is reachable, serves queries, and holds the `ingestlib`
database. The full text2SQL path lives in the sources e2e (it needs the LLM);
this just proves the SQL server itself is up and correctly provisioned.

Opt-in via RUN_INFRA_E2E=1; skips if mysql is unreachable.
"""
import os
from urllib.parse import urlsplit

import pytest

from tests.infra._util import infra_e2e

pytestmark = infra_e2e


def test_mysql_reachable_with_database():
    pymysql = pytest.importorskip("pymysql")
    dsn = os.environ.get("SQL_MYSQL_DSN", "mysql+pymysql://root:pw@localhost:3306/ingestlib")
    parts = urlsplit(dsn)
    try:
        conn = pymysql.connect(
            host=parts.hostname, port=parts.port or 3306,
            user=parts.username, password=parts.password or "",
        )
    except Exception as exc:
        pytest.skip(f"mysql unreachable: {exc}")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
            cur.execute("SHOW DATABASES LIKE 'ingestlib'")
            assert cur.fetchone(), "ingestlib database missing"
    finally:
        conn.close()
