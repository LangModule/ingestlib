"""SqlSource — generate / guard / execute / render / self-correct, against a real
SQLite DB with the LLM calls stubbed (SQL generation via achat_structured,
verified-match embeddings via aembed_text). SQLite is serverless and no model is
hit, so the whole control flow runs ungated and deterministically."""
import pytest

from ingestlib.sources.sql import source as source_mod
from ingestlib.sources.sql.safety import UnsafeQuery
from ingestlib.sources.sql.source import SqlSource


def stub_achat(monkeypatch, sql_queue=None, params=None):
    """Fake achat_structured: pops the next SQL for generation calls, and fills
    verified params from `params` for the dynamic param-model calls."""
    state = {"prompts": [], "calls": 0}
    queue = list(sql_queue or [])

    async def fake(prompt, model, **kw):
        state["prompts"].append(prompt)
        state["calls"] += 1
        fields = set(model.model_fields)
        if fields == {"sql"}:                       # _GeneratedSQL
            return model(sql=queue.pop(0))
        return model(**{n: (params or {}).get(n, "x") for n in fields})  # VerifiedParams

    monkeypatch.setattr(source_mod, "achat_structured", fake)
    return state


def stub_aembed(monkeypatch):
    """Fake aembed_text: a keyword-basis vector, so a question and a verified
    description that share a trigger word land on the same axis (cosine 1.0) and
    unrelated text is orthogonal (cosine 0)."""
    keys = ("ready", "status", "revenue")

    async def fake(text, **kw):
        t = (text or "").lower()
        v = [1.0 if k in t else 0.0 for k in keys]
        v.append(0.0 if any(v) else 1.0)            # a distinct "other" axis
        return v

    monkeypatch.setattr(source_mod, "aembed_text", fake)


# ---- health ----

async def test_health_ok_against_real_sqlite(rx_spec):
    status, detail = await SqlSource(rx_spec()).health()
    assert status == "ok" and "sqlite" in detail


async def test_health_fails_on_an_unreachable_db(rx_spec):
    src = SqlSource(rx_spec(dsn="sqlite:////no/such/dir/x.db"))
    status, detail = await src.health()
    assert status == "fail" and "rx" in detail


# ---- generation path ----

async def test_generate_executes_and_renders(rx_spec, monkeypatch):
    stub_achat(monkeypatch, sql_queue=["SELECT rx_id, status FROM rx WHERE status='ready'"])
    [r] = await SqlSource(rx_spec()).answer("which are ready?")
    assert r.source == "rx" and r.source_type == "structured"
    assert r.provenance["verified"] is False
    assert "rx_id | status" in r.content            # rendered header
    assert "ready" in r.content
    assert r.raw["columns"] == ["rx_id", "status"]
    assert len(r.raw["rows"]) == 2


async def test_generated_sql_gets_a_limit_injected(rx_spec, monkeypatch):
    stub_achat(monkeypatch, sql_queue=["SELECT * FROM rx"])
    [r] = await SqlSource(rx_spec(row_limit=1)).answer("all rows")
    assert "LIMIT 1" in r.provenance["sql"]
    assert len(r.raw["rows"]) == 1                   # the row cap actually bit


async def test_disallowed_generated_statement_is_rejected(rx_spec, monkeypatch):
    stub_achat(monkeypatch, sql_queue=["DELETE FROM rx"])
    with pytest.raises(UnsafeQuery, match="delete"):
        await SqlSource(rx_spec()).answer("delete everything")


# ---- self-correction ----

async def test_self_corrects_once_on_execution_error(rx_spec, monkeypatch):
    state = stub_achat(monkeypatch, sql_queue=[
        "SELECT nope FROM rx",                        # bad column → OperationalError
        "SELECT rx_id FROM rx WHERE status='ready'",  # the corrected query
    ])
    [r] = await SqlSource(rx_spec()).answer("ready ids?")
    assert state["calls"] == 2                        # generated, then re-generated
    assert "ERROR" in state["prompts"][1]             # the retry fed the error back
    assert len(r.raw["rows"]) == 2


async def test_self_correct_gives_up_after_one_retry(rx_spec, monkeypatch):
    state = stub_achat(monkeypatch, sql_queue=["SELECT bad FROM rx", "SELECT worse FROM rx"])
    with pytest.raises(Exception):
        await SqlSource(rx_spec()).answer("q")
    assert state["calls"] == 2                        # one retry only, then it propagates


# ---- verified queries ----

async def test_verified_query_runs_without_generation(rx_spec, monkeypatch):
    verified = {"ready_count": {
        "description": "how many prescriptions are ready",
        "sql": "SELECT COUNT(*) AS n FROM rx WHERE status='ready'",
    }}
    stub_aembed(monkeypatch)
    gen = stub_achat(monkeypatch)                     # must NOT be called to generate
    [r] = await SqlSource(rx_spec(verified=verified)).answer("how many are ready?")
    assert r.provenance["verified"] is True
    assert r.provenance["sql"].startswith("SELECT COUNT(*)")
    assert gen["calls"] == 0                          # no params → no LLM call at all
    assert r.raw["rows"] == [(2,)]


async def test_verified_query_fills_params(rx_spec, monkeypatch):
    verified = {"by_status": {
        "description": "prescriptions with a given status",
        "sql": "SELECT rx_id FROM rx WHERE status = :status",
        "params": ["status"],
    }}
    stub_aembed(monkeypatch)
    stub_achat(monkeypatch, params={"status": "ready"})
    [r] = await SqlSource(rx_spec(verified=verified)).answer("show the status prescriptions")
    assert r.provenance["verified"] is True
    assert r.provenance["params"] == {"status": "ready"}
    assert {row[0] for row in r.raw["rows"]} == {1, 2}


async def test_no_verified_match_falls_back_to_generation(rx_spec, monkeypatch):
    verified = {"ready_count": {"description": "ready prescriptions", "sql": "SELECT 999"}}
    stub_aembed(monkeypatch)
    gen = stub_achat(monkeypatch, sql_queue=["SELECT rx_id FROM rx"])
    # an unrelated question shares no trigger word → cosine 0 → below threshold
    [r] = await SqlSource(rx_spec(verified=verified)).answer("an unrelated question")
    assert r.provenance["verified"] is False
    assert gen["calls"] == 1
    assert r.raw["rows"] and 999 not in {row[0] for row in r.raw["rows"]}


# ---- rendering + schema ----

def test_render_empty_and_header(rx_spec):
    src = SqlSource(rx_spec())
    assert src._render(["a", "b"], []) == "(no rows)"
    out = src._render(["a", "b"], [(1, None), (2, "x")])
    lines = out.splitlines()
    assert lines[0] == "a | b"
    assert lines[1] == "1 | "                         # None → empty cell
    assert lines[2] == "2 | x"


async def test_schema_introspection_includes_tables_and_hints(rx_spec):
    schema = await SqlSource(rx_spec())._schema()
    assert "TABLE rx" in schema
    assert "rx_id" in schema and "status" in schema
    assert "status is ready|pending" in schema        # the tables hint carried through
