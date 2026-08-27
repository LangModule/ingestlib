"""Schema-RAG e2e — two things a stubbed test can't prove.

Part A (per-dialect introspection, no model): foreign-key reflection, card
building, and FK-graph closure differ by dialect (get_foreign_keys, type names,
sample-value fetch), so each supported backend is exercised on a real wide schema
with genuine FKs. This needs only the database, not an LLM — so SQLite and DuckDB
run ungated in `make test`, Postgres when RUN_PGVECTOR_E2E + PGVECTOR_URL are set,
MySQL when SQL_MYSQL_DSN is set.

Part B (retrieval actually helps, RUN_SQL_E2E): on a wide schema (core tables +
distractor tables with ambiguous columns), retrieval must shrink the schema
prompt AND still select the tables needed to answer — real embeddings, real
generated SQL. This is the regression guard behind the scratchpad A/B that first
demonstrated the width limitation (kept modest here so the one-time embed stays
under provider rate limits — see _DISTRACT).
"""
import os

import pytest

pytest.importorskip("sqlalchemy")

# ---- a portable wide schema with real foreign keys ----

# Table-level FOREIGN KEY constraints (not inline REFERENCES) so the FK is created
# on MySQL/InnoDB too, not just sqlite/duckdb/postgres. customers <- orders <-
# order_items -> products -> categories; audit_log is an FK island.
_CORE_DDL = [
    "CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name VARCHAR(64))",
    "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name VARCHAR(64))",
    "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name VARCHAR(64), "
    "category_id INTEGER, price DECIMAL(10,2), "
    "FOREIGN KEY (category_id) REFERENCES categories(category_id))",
    "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, "
    "status VARCHAR(16), FOREIGN KEY (customer_id) REFERENCES customers(customer_id))",
    "CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, order_id INTEGER, "
    "product_id INTEGER, quantity INTEGER, price DECIMAL(10,2), "
    "FOREIGN KEY (order_id) REFERENCES orders(order_id), "
    "FOREIGN KEY (product_id) REFERENCES products(product_id))",
    "CREATE TABLE audit_log (id INTEGER PRIMARY KEY, action VARCHAR(64))",
]
_CORE_SEED = [
    "INSERT INTO categories VALUES (1, 'books'), (2, 'toys')",
    "INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob')",
    "INSERT INTO products VALUES (1, 'novel', 1, 12.50), (2, 'blocks', 2, 8.00)",
    "INSERT INTO orders VALUES (1, 1, 'completed'), (2, 2, 'pending')",
    "INSERT INTO order_items VALUES (1, 1, 1, 2, 12.50), (2, 1, 2, 1, 8.00)",
]
# Distractor tables, each carrying status/amount/customer_id so column names
# collide across the schema — the ambiguity that misled generation at width. Kept
# modest (not the scratchpad A/B's 64) so the one-time card embed stays well under
# provider rate limits; the point here is that retrieval prunes, not raw scale.
_DISTRACT = [f"dt_{i:02d}" for i in range(18)]


def _seed_wide(dsn: str, *, wide: bool) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.begin() as conn:
        for name in ["order_items", "orders", "products", "customers", "categories",
                     "audit_log", *_DISTRACT]:
            conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
        for stmt in _CORE_DDL + _CORE_SEED:
            conn.execute(text(stmt))
        if wide:
            for name in _DISTRACT:
                conn.execute(text(
                    f"CREATE TABLE {name} (id INTEGER PRIMARY KEY, customer_id INTEGER, "
                    f"status VARCHAR(16), amount DECIMAL(10,2), note VARCHAR(64))"
                ))
                conn.execute(text(
                    f"INSERT INTO {name} VALUES (1, 1, 'pending', 5.00, 'n')"
                ))
    engine.dispose()


def _drop_wide(dsn: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    with engine.begin() as conn:
        for name in ["order_items", "orders", "products", "customers", "categories",
                     "audit_log", *_DISTRACT]:
            conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
    engine.dispose()


# ---- Part A: per-dialect FK introspection + closure (no LLM) ----

async def _assert_fk_introspection(dsn: str):
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.sources.sql.schema import SchemaIndex

    reset_engines()
    try:
        idx = SchemaIndex(dsn)
        await idx.ensure_cards()                          # introspect only — no embedding
        assert idx.table_count >= 6
        card = idx._cards["order_items"].render()
        assert "-> orders.order_id" in card, f"FK not reflected on this dialect:\n{card}"
        assert "-> products.product_id" in card
        # customers and products connect only through orders -> order_items
        closed = idx.fk_closure({"customers", "products"})
        assert {"customers", "products", "orders", "order_items"} <= closed
        assert "audit_log" not in idx.fk_closure({"customers", "audit_log"}) - {"audit_log"}
    finally:
        reset_engines()


async def test_sqlite_fk_introspection(tmp_path):
    dsn = f"sqlite:///{tmp_path / 'wide.db'}"
    _seed_wide(dsn, wide=False)
    await _assert_fk_introspection(dsn)


async def test_duckdb_fk_introspection(tmp_path):
    pytest.importorskip("duckdb_engine")
    dsn = f"duckdb:///{tmp_path / 'wide.duckdb'}"
    _seed_wide(dsn, wide=False)
    await _assert_fk_introspection(dsn)


@pytest.mark.skipif(
    os.environ.get("RUN_PGVECTOR_E2E") != "1",
    reason="postgres path is opt-in: RUN_PGVECTOR_E2E=1 + PGVECTOR_URL",
)
async def test_postgres_fk_introspection():
    from ingestlib.config import get_config

    # PGVECTOR_URL comes from .env, not the shell — an earlier reset_config() in the
    # suite un-sets dotenv keys, so re-load before reading (mirrors the snowflake e2e).
    get_config()
    url = os.environ.get("PGVECTOR_URL")
    if not url:
        pytest.skip("PGVECTOR_URL not set in .env")
    dsn = url.replace("postgresql://", "postgresql+psycopg://", 1)
    _seed_wide(dsn, wide=False)
    try:
        await _assert_fk_introspection(dsn)
    finally:
        _drop_wide(dsn)


@pytest.mark.skipif(
    not os.environ.get("SQL_MYSQL_DSN"),
    reason="set SQL_MYSQL_DSN=mysql+pymysql://root:pw@localhost:3306/ingestlib (compose `mysql`)",
)
async def test_mysql_fk_introspection():
    dsn = os.environ["SQL_MYSQL_DSN"]
    _seed_wide(dsn, wide=False)
    try:
        await _assert_fk_introspection(dsn)
    finally:
        _drop_wide(dsn)


# ---- Part B: retrieval shrinks the prompt and keeps the right tables (real model) ----

@pytest.mark.skipif(
    os.environ.get("RUN_SQL_E2E") != "1",
    reason="sql e2e is opt-in: set RUN_SQL_E2E=1 (needs the configured LLM + embeddings)",
)
async def test_schema_rag_shrinks_prompt_and_keeps_needed_tables(tmp_path):
    from ingestlib.config import SourceSpec
    from ingestlib.sources.sql.engine import reset_engines
    from ingestlib.sources.sql.source import SqlSource

    dsn = f"sqlite:///{tmp_path / 'wide.db'}"
    _seed_wide(dsn, wide=True)
    reset_engines()
    try:
        rag = SqlSource(SourceSpec(
            name="shop", type="sqlite", dsn=dsn, row_limit=100,
            schema_rag="on", schema_rag_top_k=6,
            tables={"order_items": "one row per line item; price is the unit price"},
        ))
        question = "what is the total revenue across all order line items?"

        index = rag._schema_index()
        await index.ensure_cards()
        full = index.serialize_all()
        pruned = await rag._schema(question)

        # deterministic: the retrieved schema is smaller yet keeps the line-item table
        assert len(pruned) < len(full), "retrieval should shrink the schema prompt"
        assert "TABLE order_items" in pruned
        assert pruned.count("TABLE ") < full.count("TABLE "), "retrieval must prune tables"
        assert pruned.count("TABLE ") <= 16, "should not pull in most of the schema"

        # live: the generated SQL still answers, and reads from order_items
        [r] = await rag.answer(question)
        assert r.provenance["verified"] is False
        assert "order_items" in r.provenance["sql"].lower()
        assert r.raw["rows"], "expected a revenue figure from generated SQL"
    finally:
        reset_engines()
