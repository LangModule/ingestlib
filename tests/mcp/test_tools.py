"""MCP tool functions + the JSON-Schema→Pydantic glue — pure, always run.

Services/operations are stubbed at the tools module's seams; the tests pin
the tool contract: bounded JSON out, the exact fields an agent sees, the
read/write split, and the schema conversion that makes extract work over MCP.
"""
from pydantic import BaseModel

import pytest

import ingestlib.mcp.tools as tools
from ingestlib.mcp.schema import model_from_json_schema


# ---------- JSON Schema → Pydantic ----------


def test_schema_scalars_and_required():
    schema = {
        "type": "object",
        "title": "Invoice",
        "properties": {
            "number": {"type": "string", "description": "the invoice number"},
            "total": {"type": "number"},
            "paid": {"type": "boolean"},
        },
        "required": ["number"],
    }
    Model = model_from_json_schema("X", schema)
    assert issubclass(Model, BaseModel)
    assert Model.__name__ == "Invoice"
    inst = Model(number="INV-1", total=9.5, paid=True)
    assert inst.number == "INV-1" and inst.total == 9.5
    # required field enforced
    with pytest.raises(Exception):
        Model(total=1.0)
    # optional fields default to None
    assert Model(number="INV-2").total is None


def test_schema_array_and_nested():
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
            "vendor": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        },
    }
    Model = model_from_json_schema("Doc", schema)
    inst = Model(tags=["a", "b"], vendor={"name": "Acme"})
    assert inst.tags == ["a", "b"]
    assert inst.vendor.name == "Acme"


def test_schema_rejects_non_object():
    with pytest.raises(ValueError, match="JSON Schema object"):
        model_from_json_schema("X", {"type": "string"})


# ---------- tools ----------


async def test_search_returns_bounded_cited_hits(monkeypatch):
    from ingestlib.services.retrieve.models import Hit, RetrievalResult
    from ingestlib.storage.base import RetrievedChunk

    long_text = "x" * 5000
    hit = Hit(
        chunk=RetrievedChunk(score=0.9, document_id="d" * 64, chunk_id=0,
                             section="methods", heading="Recruitment",
                             text=long_text, pages=[4]),
        vector_score=0.8, rerank_score=0.95,
    )
    captured = {}

    async def fake(question, **kw):
        captured.update(kw)
        return RetrievalResult(question=question, hits=[hit])

    monkeypatch.setattr(tools, "aretrieve", fake)
    out = await tools.search("how were participants recruited?", top_k=3, namespace="ns")

    assert out["hit_count"] == 1
    h = out["hits"][0]
    assert h["rank"] == 1 and h["citation"].startswith("doc ")
    assert h["section"] == "methods" and h["pages"] == [4]
    assert len(h["snippet"]) <= 401 and h["snippet"].endswith("…")  # clipped
    assert captured["top_k"] == 3 and captured["namespace"] == "ns"


async def test_ingest_maps_result_and_checks_file(monkeypatch, tmp_path):
    from ingestlib.services.ingest.models import IngestResult

    async def fake(path, **kw):
        return IngestResult(status="replaced", doc_id="abc", filename="r.pdf",
                            category="report", chunks=5, vectors=5,
                            replaced_doc_id="old123")

    monkeypatch.setattr(tools, "aingest", fake)
    doc = tmp_path / "r.pdf"
    doc.write_bytes(b"x")
    out = await tools.ingest(str(doc))
    assert out["status"] == "replaced" and out["replaced_doc_id"] == "old123"

    with pytest.raises(ValueError, match="file not found"):
        await tools.ingest(str(tmp_path / "nope.pdf"))


async def test_extract_builds_schema_and_shapes_items(monkeypatch, tmp_path):
    from ingestlib.operations.extract.models import ExtractedItem, ExtractResult, FieldValue

    class _Fake(BaseModel):
        total: float

    captured = {}

    async def fake(source, schema, **kw):
        captured["schema"] = schema
        captured.update(kw)
        item = ExtractedItem(
            value=_Fake(total=20.0),
            fields={"total": FieldValue(confidence=0.9, pages=[10], grounded=True)},
            pages=[10],
        )
        return ExtractResult(items=[item], schema_name="ExtractSchema", mode="many", pages_used=16)

    monkeypatch.setattr(tools, "aextract", fake)
    doc = tmp_path / "receipt.pdf"
    doc.write_bytes(b"x")

    out = await tools.extract(
        str(doc),
        {"type": "object", "properties": {"total": {"type": "number"}}},
        mode="many",
    )
    # the agent's JSON schema became a real Pydantic model passed to extract
    assert issubclass(captured["schema"], BaseModel)
    assert captured["mode"] == "many"
    assert out["item_count"] == 1
    item = out["items"][0]
    assert item["value"] == {"total": 20.0}
    assert item["fields"]["total"] == {"grounded": True, "confidence": 0.9, "pages": [10]}


async def test_list_documents_namespace_filter(monkeypatch):
    from ingestlib.storage.artifacts import DocumentMeta

    docs = [
        DocumentMeta(doc_id="a", filename="a.pdf", namespace="", page_count=3, chunks=4),
        DocumentMeta(doc_id="b", filename="b.pdf", namespace="tenant", page_count=1, chunks=1),
    ]
    monkeypatch.setattr(tools.artifacts, "list_documents", lambda: docs)

    allns = await tools.list_documents()
    assert allns["count"] == 2
    scoped = await tools.list_documents(namespace="tenant")
    assert scoped["count"] == 1 and scoped["documents"][0]["doc_id"] == "b"


async def test_sync_reports_counts_and_actions(monkeypatch):
    from ingestlib.services.lifecycle.models import SyncAction, SyncResult

    async def fake(directory, **kw):
        return SyncResult(directory=directory, dry_run=kw.get("dry_run", False),
                          actions=[SyncAction(path="/c/new.pdf", action="ingest"),
                                   SyncAction(path="/c/old.pdf", action="prune", doc_id="x")])

    monkeypatch.setattr(tools, "async_sync", fake)
    out = await tools.sync("/c", prune=True, dry_run=True)
    assert out["dry_run"] is True
    assert out["counts"] == {"ingest": 1, "prune": 1}
    assert out["actions"][0]["action"] == "ingest"


async def test_remove_and_backfill(monkeypatch):
    from ingestlib.services.lifecycle.models import BackfillResult, RemoveResult

    async def fake_remove(target, **kw):
        return RemoveResult(doc_id="d", filename="f.pdf", vectors_deleted=3, artifacts_deleted=7)

    async def fake_backfill(**kw):
        return BackfillResult(documents=2, chunks=9, skipped=["z"])

    monkeypatch.setattr(tools, "aremove", fake_remove)
    monkeypatch.setattr(tools, "abackfill", fake_backfill)
    assert (await tools.remove("f.pdf"))["vectors_deleted"] == 3
    b = await tools.backfill()
    assert b["documents"] == 2 and b["skipped"] == 1


async def test_doctor_collects_check_statuses(monkeypatch):
    from ingestlib.cli import doctor as doc

    monkeypatch.setattr(doc, "check_python", lambda: ("ok", "python 3.13"))
    monkeypatch.setattr(doc, "check_llm", lambda: ("fail", "dead server"))
    for name in ("check_libreoffice", "check_ocr_server", "check_embeddings",
                 "check_reranker", "check_artifact_store", "check_vector_store"):
        monkeypatch.setattr(doc, name, lambda: ("ok", "fine"))

    out = await tools.doctor()
    assert out["healthy"] is False  # the llm fail flips it
    by = {c["check"]: c["status"] for c in out["checks"]}
    assert by["python"] == "ok" and by["llm"] == "fail"


def test_read_write_tool_split():
    names = {t.__name__ for t in tools.ALL_TOOLS}
    assert names == {"search", "ingest", "extract", "classify", "list_documents",
                     "remove", "sync", "backfill", "doctor"}
    assert tools.WRITE_TOOLS == {"ingest", "remove", "sync", "backfill"}
    # read-only tools are everything not in WRITE_TOOLS
    assert "extract" not in tools.WRITE_TOOLS and "search" not in tools.WRITE_TOOLS
