"""`ingestlib describe-schema` through main() — introspects a real SQLite DB and
prints a ready-to-paste `tables:` block, with the LLM stubbed at the CLI seam. No
model is hit and SQLite is serverless, so this runs ungated."""
import pytest

pytest.importorskip("sqlalchemy")

from ingestlib.cli import main


@pytest.fixture()
def shop_source(tmp_path, monkeypatch):
    """A real SQLite DB with two tables, registered as a SQL source, with
    achat_structured stubbed to echo the table name into a description."""
    from sqlalchemy import create_engine, text

    import ingestlib.config as config_mod
    import ingestlib.foundations.llm as llm_mod
    from ingestlib.config import SourceSpec, SourcesConfig
    from ingestlib.sources.sql.engine import reset_engines

    db = tmp_path / "shop.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("CREATE TABLE orders (order_id INTEGER PRIMARY KEY, total REAL)"))
        conn.execute(text("INSERT INTO customers VALUES (1,'Alice')"))
        conn.execute(text("INSERT INTO orders VALUES (1,42.0)"))
    engine.dispose()
    reset_engines()

    spec = SourceSpec(name="shop", type="sqlite", dsn=f"sqlite:///{db}")
    monkeypatch.setattr(
        config_mod, "get_sources_config", lambda: SourcesConfig(sources={"shop": spec})
    )

    async def fake_achat(prompt, model, **kw):
        # the prompt carries the rendered card; name the table it describes
        name = "customers" if "TABLE customers" in prompt else "orders"
        return model(description=f"rows of {name}")

    monkeypatch.setattr(llm_mod, "achat_structured", fake_achat)
    yield
    reset_engines()


def test_describe_schema_prints_tables_block(shop_source, capsys):
    assert main(["describe-schema", "shop"]) == 0
    out = capsys.readouterr().out
    assert "tables:" in out
    assert "customers: rows of customers" in out
    assert "orders: rows of orders" in out
    assert "REVIEW before trusting" in out


def test_describe_schema_writes_out_file(shop_source, tmp_path, capsys):
    dest = tmp_path / "hints.yaml"
    assert main(["describe-schema", "shop", "--out", str(dest)]) == 0
    assert "customers: rows of customers" in dest.read_text()
    assert "wrote" in capsys.readouterr().out


def test_describe_schema_rejects_unknown_source(shop_source, capsys):
    assert main(["describe-schema", "nope"]) == 1
    assert "unknown source" in capsys.readouterr().out


def test_describe_schema_skips_a_failing_table(shop_source, capsys, monkeypatch):
    """A throttled/failed LLM call on one table is skipped, not fatal — the rest
    of the (possibly wide, cryptic) schema still gets described."""
    import ingestlib.foundations.llm as llm_mod

    async def flaky_achat(prompt, model, **kw):
        if "TABLE orders" in prompt:
            raise RuntimeError("simulated throttle")
        return model(description="rows of customers")

    monkeypatch.setattr(llm_mod, "achat_structured", flaky_achat)
    assert main(["describe-schema", "shop"]) == 0           # must not crash
    out = capsys.readouterr().out
    assert "customers: rows of customers" in out             # survivor kept
    assert "orders:" not in out                              # failed table omitted
