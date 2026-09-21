"""`ingestlib eval-sql` — the pure scoring (matched/summarize) and the command
driven through main() against a real SQLite SqlSource with the LLM stubbed. No
model is hit and SQLite is serverless, so this runs ungated."""
import pytest

pytest.importorskip("sqlalchemy")

from ingestlib.cli import main
from ingestlib.cli.eval_sql import matched, summarize


# ---- scoring (pure) ----

def test_matched_numbers_are_exact():
    assert matched(5, [(5,)]) is True
    assert matched(5, [(25,)]) is False          # "5" must not match "25"
    assert matched("1,000", [(1000,)]) is True    # comma-insensitive


def test_matched_text_is_case_insensitive_substring():
    assert matched("Asia", [("ASIA PACIFIC",)]) is True
    assert matched("europe", [("Africa",)]) is False


def test_matched_list_hits_on_any():
    assert matched(["asia", "europe"], [("EUROPE",)]) is True
    assert matched(["asia", "europe"], [("antarctica",)]) is False


def test_summarize_splits_generated_and_verified():
    rows = [
        {"path": "generated", "ran": True, "hit": True},
        {"path": "generated", "ran": True, "hit": False},
        {"path": "verified", "ran": True, "hit": True},
        {"path": "error", "ran": False, "hit": False},
    ]
    s = summarize(rows)
    assert s["questions"] == 4
    assert s["match_rate_generated"] == 0.5
    assert s["match_rate_verified"] == 1.0
    assert s["error_count"] == 1


# ---- command through main() ----

@pytest.fixture()
def rx_eval_source(tmp_path, monkeypatch):
    """A real SQLite rx DB registered as a SQL source, with SQL generation stubbed
    to a valid COUNT query, plus a dataset file."""
    from sqlalchemy import create_engine, text

    import ingestlib.foundations.llm as llm_mod
    import ingestlib.sources.registry as registry_mod
    from ingestlib.config import SourceSpec, SourcesConfig
    from ingestlib.sources.registry import reset_registry
    from ingestlib.sources.sql.engine import reset_engines

    db = tmp_path / "rx.db"
    engine = create_engine(f"sqlite:///{db}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE rx (rx_id INTEGER PRIMARY KEY, status TEXT)"))
        conn.execute(text("INSERT INTO rx VALUES (1,'ready'),(2,'ready'),(3,'pending')"))
    engine.dispose()
    reset_engines()
    reset_registry()

    spec = SourceSpec(name="rx", type="sqlite", dsn=f"sqlite:///{db}", schema_rag="off")
    # the registry binds get_sources_config at import; patch its reference
    monkeypatch.setattr(
        registry_mod, "get_sources_config", lambda: SourcesConfig(sources={"rx": spec})
    )

    async def fake_achat(prompt, model, **kw):
        return model(sql="SELECT count(*) AS n FROM rx WHERE status = 'ready'")

    monkeypatch.setattr(llm_mod, "achat_structured", fake_achat)

    dataset = tmp_path / "rx_eval.yaml"
    dataset.write_text(
        "questions:\n"
        "  - id: ready_count\n"
        "    question: how many prescriptions are ready?\n"
        "    expect: 2\n"
    )
    yield str(dataset)
    reset_engines()
    reset_registry()


def test_eval_sql_reports_a_hit(rx_eval_source, capsys):
    assert main(["eval-sql", "rx", "--dataset", rx_eval_source]) == 0
    out = capsys.readouterr().out
    assert "match rate (overall)" in out
    assert "100%" in out
    assert "✓" in out


def test_eval_sql_missing_dataset_errors(capsys, monkeypatch):
    import ingestlib.config as config_mod
    from ingestlib.config import SourceSpec, SourcesConfig

    spec = SourceSpec(name="rx", type="sqlite", dsn="sqlite:///x.db")
    monkeypatch.setattr(
        config_mod, "get_sources_config", lambda: SourcesConfig(sources={"rx": spec})
    )
    assert main(["eval-sql", "rx", "--dataset", "/no/such/dataset.yaml"]) == 1
    assert "no eval dataset" in capsys.readouterr().out
