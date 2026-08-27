"""SchemaIndex — card building, embedding retrieval, and FK-graph closure against
a real SQLite DB with a genuine foreign-key topology. SQLite is serverless and the
embedding function is stubbed with a deterministic keyword-basis vector, so the
whole schema-RAG control flow runs ungated in `make test` — no model is hit and
retrieval is reproducible. The one live piece (real embeddings on wide schemas)
is exercised by the gated e2e suite, not here."""
import hashlib

import pytest

pytest.importorskip("sqlalchemy")

from ingestlib.sources.sql import engine as eng
from ingestlib.sources.sql.schema import SchemaIndex, _bridge, _fk_graph

# customers <- orders <- order_items -> products -> categories ; payments -> orders.
# audit_log and settings are islands (no FK), so they must never be dragged in by
# closure. This is the canonical shape for the bridge-table test.
_DDL = [
    "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, email TEXT)",
    "CREATE TABLE categories (category_id INTEGER PRIMARY KEY, name TEXT)",
    "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, "
    "category_id INTEGER REFERENCES categories(category_id), price REAL)",
    "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, "
    "customer_id INTEGER REFERENCES customers(customer_id), status TEXT, total REAL)",
    "CREATE TABLE order_items (item_id INTEGER PRIMARY KEY, "
    "order_id INTEGER REFERENCES orders(order_id), "
    "product_id INTEGER REFERENCES products(product_id), quantity INTEGER, price REAL)",
    "CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, "
    "order_id INTEGER REFERENCES orders(order_id), status TEXT, amount REAL)",
    "CREATE TABLE audit_log (id INTEGER PRIMARY KEY, action TEXT)",
    "CREATE TABLE settings (id INTEGER PRIMARY KEY, key TEXT, value TEXT)",
]
_SEED = [
    "INSERT INTO customers VALUES (1,'Alice','a@x.com'),(2,'Bob','b@x.com')",
    "INSERT INTO orders VALUES (1,1,'completed',100),(2,2,'pending',50)",
]
_ALL = {
    "customers", "categories", "products", "orders",
    "order_items", "payments", "audit_log", "settings",
}


@pytest.fixture()
def shop_dsn(tmp_path):
    from sqlalchemy import create_engine, text

    db = tmp_path / "shop.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        for stmt in _DDL + _SEED:
            conn.execute(text(stmt))
    engine.dispose()
    eng.reset_engines()
    yield f"sqlite:///{db}"
    eng.reset_engines()


def stub_embed(keys):
    """A keyword-basis embedder: the vector marks which of `keys` appear in the
    text, plus a distinct 'other' axis so unrelated text is orthogonal (cosine 0).
    A question and a card that share a table name land on the same axis."""

    async def fake(text, **kw):
        t = (text or "").lower()
        v = [1.0 if k in t else 0.0 for k in keys]
        v.append(0.0 if any(v) else 1.0)
        return v

    return fake


def _index(dsn, **kw):
    keys = ("customers", "orders", "order_items", "products", "categories", "payments")
    kw.setdefault("embed_pace", 0.0)          # stub embedder never throttles — no pacing
    return SchemaIndex(dsn, embed=stub_embed(keys), **kw)


# ---- card content (M-Schema contract) ----

async def test_cards_carry_types_pk_fk_and_samples(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    card = idx._cards["order_items"].render()
    assert card.startswith("TABLE order_items")           # downstream substring checks hold
    assert "quantity INTEGER" in card                     # column + type
    assert "item_id INTEGER  PK" in card                  # primary-key marker
    assert "-> orders.order_id" in card                   # foreign-key arrow
    orders = idx._cards["orders"].render()
    assert "e.g. 'completed'" in orders or "e.g. 'pending'" in orders  # sampled values


async def test_hint_is_rendered_as_a_comment(shop_dsn):
    idx = _index(shop_dsn, table_hints={"orders": "one row per placed order"})
    await idx.build()
    assert "-- one row per placed order" in idx._cards["orders"].render()


async def test_build_is_idempotent(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    vectors = idx._vectors
    await idx.build()                                     # second call is a no-op
    assert idx._vectors is vectors
    assert idx.table_count == len(_ALL)


# ---- retrieval ----

async def test_retrieve_selects_relevant_tables(shop_dsn):
    idx = _index(shop_dsn)
    picked = await idx.retrieve("how many orders and customers do we have", top_k=2)
    assert {"customers", "orders"} <= picked
    assert "audit_log" not in picked and "settings" not in picked


async def test_retrieve_pulls_bridge_tables_via_fk_closure(shop_dsn):
    # "revenue from products" embeds to products (+ order_items via its card text);
    # closure must connect products to any order table through the join chain.
    idx = _index(shop_dsn)
    picked = await idx.retrieve("total revenue from products sold in order_items", top_k=2)
    assert {"products", "order_items"} <= picked
    # order_items already bridges to products directly; the island tables stay out.
    assert "audit_log" not in picked


async def test_serialize_is_stable_introspection_order(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    block = idx.serialize({"payments", "customers", "orders"})
    # rendered in introspection order (customers, orders, payments), not set order
    at = {t: block.index(f"TABLE {t}") for t in ("customers", "orders", "payments")}
    assert at["customers"] < at["orders"] < at["payments"]


async def test_serialize_empty_selection(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    assert idx.serialize(set()) == "(no tables)"


# ---- FK graph + closure (pure) ----

async def test_fk_closure_adds_multi_hop_bridge(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    # customers and products are connected only through orders -> order_items
    closed = idx.fk_closure({"customers", "products"})
    assert {"customers", "products", "orders", "order_items"} <= closed


async def test_fk_closure_no_bridge_for_islands(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    closed = idx.fk_closure({"customers", "audit_log"})
    assert closed == {"customers", "audit_log"}           # unreachable → nothing added


async def test_fk_closure_direct_neighbors_need_no_bridge(shop_dsn):
    idx = _index(shop_dsn)
    await idx.build()
    assert idx.fk_closure({"customers", "orders"}) == {"customers", "orders"}


# ---- scale: retrieval + closure out of 150+ tables ----

@pytest.fixture()
def big_dsn(tmp_path):
    """The core schema plus 144 distractor tables (150+ total) — the wide-schema
    case schema-RAG exists for. Distractors carry colliding column names so a
    dump-all prompt would be a mess; retrieval must still return a small subset."""
    from sqlalchemy import create_engine, text

    db = tmp_path / "big.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON"))
        for stmt in _DDL + _SEED:
            conn.execute(text(stmt))
        for i in range(144):
            conn.execute(text(
                f"CREATE TABLE dt_{i:03d} (id INTEGER PRIMARY KEY, customer_id INTEGER, "
                f"status TEXT, amount REAL, note TEXT)"
            ))
    engine.dispose()
    eng.reset_engines()
    yield f"sqlite:///{db}"
    eng.reset_engines()


async def test_retrieval_scales_to_150_plus_tables(big_dsn):
    idx = _index(big_dsn)
    await idx.build()
    assert idx.table_count >= 150                          # genuinely wide

    picked = await idx.retrieve("revenue from orders order_items and products", top_k=3)
    # Out of 152 tables, retrieval returns a small, join-complete subset with the
    # 144 distractors excluded. order_items (the revenue table) uniquely matches
    # all three query tokens, so it is always selected; which of the other
    # orders-linked core tables round out the top-k is a semantic call the stub
    # can't make (real embeddings do — see the live e2e), so we don't pin it here.
    assert "order_items" in picked
    assert len(picked) <= 10                               # bounded — not most of 150
    assert not any(t.startswith("dt_") for t in picked)    # 144 distractors excluded
    assert all(t in idx._cards for t in picked)

    block = idx.serialize(picked)
    assert block.count("TABLE ") == len(picked)            # prompt holds only the subset


# ---- inferred join edges (schemas with no declared FKs) ----

@pytest.fixture()
def no_fk_dsn(tmp_path):
    """The same entities but with NO foreign keys declared — the common enterprise
    case. Conventional `<entity>_id` columns reference `id` PKs; edges must be
    inferred from naming, not reflected."""
    from sqlalchemy import create_engine, text

    db = tmp_path / "nofk.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, status TEXT)"
        ))
        conn.execute(text(
            "CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER, "
            "product_id INTEGER, quantity INTEGER)"
        ))
    engine.dispose()
    eng.reset_engines()
    yield f"sqlite:///{db}"
    eng.reset_engines()


async def test_inferred_edges_connect_a_schema_without_declared_fks(no_fk_dsn):
    idx = _index(no_fk_dsn)
    await idx.ensure_cards()
    # customers and products connect only through orders -> order_items, all inferred
    closed = idx.fk_closure({"customers", "products"})
    assert {"customers", "products", "orders", "order_items"} <= closed
    # and the inference is visible to the model in the card
    assert "-> orders.id (inferred)" in idx._cards["order_items"].render()
    assert "-> customers.id (inferred)" in idx._cards["orders"].render()


async def test_declared_fk_takes_precedence_over_inferred(shop_dsn):
    idx = _index(shop_dsn)                                  # shop_dsn HAS declared FKs
    await idx.ensure_cards()
    card = idx._cards["order_items"].render()
    assert "-> orders.order_id" in card                     # declared, not "(inferred)"
    assert "(inferred)" not in card


# ---- persistence + graceful/incremental build ----

def _counting_stub(fail_for=()):
    """A stub embedder that counts calls and (optionally) raises for named tables
    (a card renders as "TABLE <name>\\n..."). Deterministic keyword-basis vector."""
    keys = ("customers", "orders", "order_items", "products", "categories", "payments")
    state = {"calls": 0}

    async def fake(text, **kw):
        state["calls"] += 1
        if any((text or "").startswith(f"TABLE {name}\n") for name in fail_for):
            raise RuntimeError("simulated throttle")
        t = (text or "").lower()
        v = [1.0 if k in t else 0.0 for k in keys]
        v.append(0.0 if any(v) else 1.0)
        return v

    return fake, state


def _persistent_index(dsn, embed, cache_dir, tag="p"):
    return SchemaIndex(dsn, embed=embed, embed_pace=0.0, cache_dir=cache_dir, cache_tag=tag)


async def test_index_persists_and_reloads_without_embedding(shop_dsn, tmp_path):
    embed1, s1 = _counting_stub()
    idx1 = _persistent_index(shop_dsn, embed1, tmp_path, tag="prov-x")
    await idx1.build()
    assert s1["calls"] == idx1.table_count                  # embedded every card once
    stem = hashlib.sha256(shop_dsn.encode()).hexdigest()[:16]
    assert (tmp_path / f"schema-{stem}.json").is_file()

    # a fresh index over the same DB + cache must NOT embed anything
    embed2, s2 = _counting_stub()
    idx2 = _persistent_index(shop_dsn, embed2, tmp_path, tag="prov-x")
    await idx2.build()
    assert s2["calls"] == 0                                 # served entirely from disk
    assert await idx2.retrieve("orders and customers", top_k=2)  # still works


async def test_cache_invalidated_by_embedding_provider_change(shop_dsn, tmp_path):
    embed1, _ = _counting_stub()
    await _persistent_index(shop_dsn, embed1, tmp_path, tag="prov-x").build()
    # a different embedding provider tag ⇒ old vectors are meaningless ⇒ rebuild
    embed2, s2 = _counting_stub()
    idx2 = _persistent_index(shop_dsn, embed2, tmp_path, tag="prov-Y")
    await idx2.build()
    assert s2["calls"] == idx2.table_count                  # re-embedded, not reused


async def test_build_skips_a_failing_card_without_crashing(shop_dsn):
    embed, _ = _counting_stub(fail_for=["payments"])
    idx = SchemaIndex(shop_dsn, embed=embed, embed_pace=0.0)
    await idx.build()                                       # must NOT raise
    assert idx._built
    assert "payments" not in idx._vectors                   # the one failure, skipped
    assert "orders" in idx._vectors                         # the rest embedded
    # a skipped table can still arrive via FK closure (payments -> orders)
    assert "payments" in idx.fk_closure({"orders", "payments"})


async def test_incremental_build_embeds_only_missing_cards(shop_dsn, tmp_path):
    # first build throttles on two tables → they are absent from the cache
    embed1, _ = _counting_stub(fail_for=["payments", "categories"])
    idx1 = _persistent_index(shop_dsn, embed1, tmp_path)
    await idx1.build()
    assert {"payments", "categories"}.isdisjoint(idx1._vectors)

    # next build loads the cached ones and embeds ONLY the two that were missing
    embed2, s2 = _counting_stub()
    idx2 = _persistent_index(shop_dsn, embed2, tmp_path)
    await idx2.build()
    assert s2["calls"] == 2                                 # just the previously-failed cards
    assert {"payments", "categories"} <= set(idx2._vectors)


def test_bridge_respects_max_hops():
    graph = {"a": {"m"}, "m": {"a", "n"}, "n": {"m", "b"}, "b": {"n"}}
    assert _bridge(graph, "a", "b", max_hops=2) == ["m", "n"]   # two intermediates
    assert _bridge(graph, "a", "b", max_hops=1) is None          # too far within bound


def test_fk_graph_is_undirected():
    from ingestlib.sources.sql.schema import TableCard

    cards = {
        "orders": TableCard("orders", [], fks=[("customer_id", "customers", "customer_id")]),
        "customers": TableCard("customers", []),
    }
    graph = _fk_graph(cards)
    assert graph["orders"] == {"customers"} and graph["customers"] == {"orders"}
