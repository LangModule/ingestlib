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
    # One table → auto mode dumps the whole (M-Schema) schema, no embedding needed.
    schema = await SqlSource(rx_spec())._schema("how many prescriptions are ready")
    assert "TABLE rx" in schema
    assert "rx_id" in schema and "status" in schema
    assert "status is ready|pending" in schema        # the tables hint carried through


# ---- schema-RAG mode switch + widen-on-error (retrieval logic itself is in test_schema.py) ----

async def test_schema_rag_off_never_retrieves(rx_spec, monkeypatch):
    """schema_rag=off dumps the whole schema and must not call the retriever."""
    from ingestlib.sources.sql.schema import SchemaIndex

    async def boom(self, *a, **kw):
        raise AssertionError("retrieve() must not run when schema_rag=off")

    monkeypatch.setattr(SchemaIndex, "retrieve", boom)
    schema = await SqlSource(rx_spec(schema_rag="off"))._schema("anything")
    assert "TABLE rx" in schema


async def test_schema_rag_auto_small_schema_dumps_without_embedding(rx_spec, monkeypatch):
    """Below the table threshold, auto mode dumps all and never embeds."""
    import ingestlib.sources.sql.schema as schema_mod

    async def boom(*a, **kw):
        raise AssertionError("a small schema must not be embedded")

    monkeypatch.setattr(schema_mod, "aembed_text", boom)
    schema = await SqlSource(rx_spec(schema_rag="auto", schema_rag_min_tables=5))._schema("q")
    assert "TABLE rx" in schema


async def test_widen_multiplies_top_k_on_retry(rx_spec, monkeypatch):
    """The self-correct retry widens the retrieval — top_k × _WIDEN_FACTOR."""
    from ingestlib.sources.sql.schema import SchemaIndex

    seen = {}

    async def fake_retrieve(self, question, *, top_k):
        seen["top_k"] = top_k
        return {"rx"}

    monkeypatch.setattr(SchemaIndex, "retrieve", fake_retrieve)
    src = SqlSource(rx_spec(schema_rag="on", schema_rag_top_k=5))
    await src._schema("q", widen=False)
    assert seen["top_k"] == 5
    await src._schema("q", widen=True)
    assert seen["top_k"] == 5 * source_mod._WIDEN_FACTOR


def test_verified_threshold_follows_embedding_provider(monkeypatch):
    """Asymmetric (bedrock) embeddings need a low floor; symmetric (openai/ollama)
    embeddings score higher, so the floor is raised to avoid over-matching."""
    import ingestlib.config as config_mod

    def _provider(name):
        cfg = type("C", (), {"embedding_provider": name})
        monkeypatch.setattr(config_mod, "get_config", lambda: cfg)

    _provider("bedrock")
    assert source_mod._verified_threshold() == source_mod._VERIFIED_THRESHOLD_ASYMMETRIC
    _provider("openai")
    assert source_mod._verified_threshold() == source_mod._VERIFIED_THRESHOLD_SYMMETRIC
    _provider("ollama")
    assert source_mod._verified_threshold() == source_mod._VERIFIED_THRESHOLD_SYMMETRIC


async def test_self_correct_retry_widens_schema(rx_spec, monkeypatch):
    """A generated query that errors once is regenerated with a widened schema,
    then succeeds — proving answer() threads widen=True into the retry."""
    stub_aembed(monkeypatch)
    widens = []

    real_schema = SqlSource._schema

    async def spy(self, question, *, widen=False):
        widens.append(widen)
        return await real_schema(self, question, widen=widen)

    monkeypatch.setattr(SqlSource, "_schema", spy)
    # first SQL hits a bogus table (errors), second is valid
    stub_achat(monkeypatch, sql_queue=[
        "SELECT count(*) FROM no_such_table",
        "SELECT count(*) FROM rx WHERE status = 'ready'",
    ])
    [result] = await SqlSource(rx_spec()).answer("how many are ready")
    assert result.raw["rows"] == [(2,)]
    assert widens == [False, True]                    # generate, then widened retry
