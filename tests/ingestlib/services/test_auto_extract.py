"""Collections (#1) — a rules.yaml collection with auto_extract runs extraction
on ingest, and the declared collection is materialized into the registry.

Uses the `pipeline` fixture; the extract SERVICE is stubbed so no LLM is called.
"""
import dataclasses
import importlib

import ingestlib.config as config_module

# services/__init__ re-exports `extract` (the function), shadowing the submodule
# on attribute lookup — grab the module object to patch its aextract.
_extract_mod = importlib.import_module("ingestlib.services.extract")


def test_auto_extract_runs_and_collection_is_materialized(pipeline, monkeypatch):
    from ingestlib.config import CollectionRule, get_config
    from ingestlib.services import ingest
    from ingestlib.storage import registry

    # classify (stubbed) tags every doc "report" — declare that collection with a
    # schema + auto_extract in the config the ingestor reads
    schema = {"type": "object", "properties": {"total": {"type": "number"}}}
    cfg = get_config()
    classify = dataclasses.replace(
        cfg.classify,
        collections={"report": CollectionRule(
            description="report", extract_schema=schema, auto_extract=True
        )},
    )
    monkeypatch.setattr(config_module, "_config", dataclasses.replace(cfg, classify=classify))

    calls: dict = {}

    async def fake_extract(source, model, *, persist=False, **kwargs):
        calls["fields"] = list(model.model_fields)
        calls["persist"] = persist
        return None

    monkeypatch.setattr(_extract_mod, "aextract", fake_extract)

    doc = pipeline.corpus / "r.pdf"
    doc.write_bytes(b"a quarterly report")
    ingest(doc, store=pipeline.store)

    # the declared collection landed in the registry with its schema
    coll = registry.get_collection("report")
    assert coll is not None
    assert coll["auto_extract"] is True
    assert coll["extract_schema"] == schema

    # auto-extract ran with the JSON-Schema bridged to a Pydantic model, persisting
    assert calls.get("persist") is True
    assert calls.get("fields") == ["total"]


def test_no_auto_extract_when_collection_lacks_flag(pipeline, monkeypatch):
    from ingestlib.config import CollectionRule, get_config
    from ingestlib.services import ingest

    cfg = get_config()
    classify = dataclasses.replace(
        cfg.classify,
        collections={"report": CollectionRule(description="report")},  # no schema/flag
    )
    monkeypatch.setattr(config_module, "_config", dataclasses.replace(cfg, classify=classify))

    ran = {"called": False}

    async def fake_extract(*a, **k):
        ran["called"] = True

    monkeypatch.setattr(_extract_mod, "aextract", fake_extract)

    doc = pipeline.corpus / "r.pdf"
    doc.write_bytes(b"a quarterly report")
    ingest(doc, store=pipeline.store)

    assert ran["called"] is False
