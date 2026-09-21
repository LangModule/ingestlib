"""Registry write path (storage/registry.py) round-trip against the live container.

Opt-in via RUN_REGISTRY_E2E=1 with the `registry` compose service up. Builds
SYNTHETIC operation results (no OCR/LLM), saves each stage, reads the whole
document back, and asserts the mapping round-trips — then checks re-ingest
overwrites in place and delete cascades.
"""
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REGISTRY_E2E") != "1",
    reason="registry e2e is opt-in: set RUN_REGISTRY_E2E=1 (needs the compose `registry` service up)",
)

DOC_ID = "write-e2e-doc"


@pytest.fixture(autouse=True)
def _clean_registry():
    from ingestlib_registry import db

    from ingestlib.cli.registry import run_registry_init
    from ingestlib.storage.registry import delete_document

    db.reset_engine()
    run_registry_init()          # guarantee the schema (siblings may drop/restore)
    delete_document(DOC_ID)      # clean slate for this doc_id
    yield
    delete_document(DOC_ID)
    db.reset_engine()


def _parse():
    from ingestlib.foundations.ocr.models import BoundingBox, Region
    from ingestlib.operations.parse.models import PageResult, ParseResult

    region = Region(
        region_type="text",
        bbox=BoundingBox(x=1.0, y=2.0, width=30.0, height=40.0),
        region_id=0,
        text="hello world",
        content="hello world",
    )
    page = PageResult(page_num=1, regions=[region], page_width=800, page_height=600)
    return ParseResult(
        pages=[page],
        source_path=Path("/tmp/report.pdf"),
        source_format="pdf",
        source_checksum=DOC_ID,
        source_metadata={"title": "Q1", "author": "Finance"},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _classify():
    from ingestlib.operations.classify.models import CategoryScore, ClassifyResult

    return ClassifyResult(
        category="invoice",
        confidence=0.9,
        reasoning="itemized charges and totals",
        alternatives=[CategoryScore(label="receipt", score=0.2)],
    )


def _split():
    from ingestlib.operations.split.models import Chunk, Section, SplitResult

    chunk = Chunk(
        chunk_id=0, section="body", heading="Totals", text="t", markdown="**Totals**",
        embedding_text="[invoice > body > Totals] **Totals**", pages=[1],
        region_ids={1: [0]}, kind="text", token_estimate=7,
    )
    section = Section(name="body", description="the body", pages=[1], chunks=[chunk])
    return SplitResult(sections=[section])


def _extract():
    from pydantic import BaseModel

    from ingestlib.operations.extract.models import ExtractedItem, ExtractResult, FieldValue

    class Invoice(BaseModel):
        total: float

    item = ExtractedItem(
        value=Invoice(total=20.0),
        fields={"total": FieldValue(confidence=0.8, region_ids={1: [0]}, pages=[1], grounded=True)},
        pages=[1],
    )
    return ExtractResult(items=[item], schema_name="Invoice", mode="one")


def _save_all():
    from ingestlib.storage import registry

    registry.save_parse(DOC_ID, _parse(), namespace="ns", source_path=Path("/tmp/report.pdf"))
    registry.save_classify(DOC_ID, _classify())
    registry.save_split(DOC_ID, _split())
    registry.save_embed(
        DOC_ID, vector_store="SqliteStore", vector_dim=1024, vector_count=1,
        embedded_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    registry.save_status(DOC_ID, status="ingested")
    registry.save_extraction(DOC_ID, _extract())


def test_full_round_trip():
    from ingestlib.storage import registry

    _save_all()
    got = registry.get_document(DOC_ID)
    assert got is not None

    doc = got["document"]
    assert doc["namespace"] == "ns"
    assert doc["source_path"] == str(Path("/tmp/report.pdf").resolve())  # save_parse resolves it
    assert doc["filename"] == "report.pdf"
    assert doc["source_format"] == "pdf"
    assert doc["source_metadata"] == {"title": "Q1", "author": "Finance"}
    assert doc["page_count"] == 1
    assert doc["section_count"] == 1
    assert doc["chunk_count"] == 1
    assert doc["status"] == "ingested"
    assert doc["category"] == "invoice"
    assert doc["classify_confidence"] == 0.9
    assert doc["classify_reasoning"] == "itemized charges and totals"
    assert doc["classify_alternatives"] == [{"label": "receipt", "score": 0.2}]
    assert doc["vector_store"] == "SqliteStore"
    assert doc["vector_dim"] == 1024
    assert doc["vector_count"] == 1
    assert doc["created_at"] == datetime(2026, 1, 1, tzinfo=timezone.utc)  # from the ParseResult
    assert doc["updated_at"] is not None                                    # set on insert

    assert [p["page_num"] for p in got["pages"]] == [1]
    assert got["pages"][0]["width"] == 800 and got["pages"][0]["height"] == 600

    assert len(got["regions"]) == 1
    region = got["regions"][0]
    assert region["region_type"] == "text"
    assert region["bbox"] == {"x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0}
    assert region["text"] == "hello world"

    assert [(s["ord"], s["name"]) for s in got["sections"]] == [(0, "body")]

    assert len(got["chunks"]) == 1
    chunk = got["chunks"][0]
    assert chunk["chunk_id"] == 0
    assert chunk["markdown"] == "**Totals**"
    assert chunk["region_ids"] == {1: [0]}   # int keys restored on read

    assert len(got["extractions"]) == 1
    extraction = got["extractions"][0]
    assert extraction["schema_name"] == "Invoice"
    assert extraction["item_index"] == 0
    assert extraction["value"] == {"total": 20.0}
    assert extraction["fields"]["total"]["grounded"] is True


def test_reingest_overwrites_in_place():
    from ingestlib.operations.split.models import Chunk, Section, SplitResult
    from ingestlib.storage import registry

    _save_all()
    # a re-split with fewer chunks must replace, not accumulate
    section = Section(name="body", description="d", pages=[1], chunks=[
        Chunk(chunk_id=0, section="body", heading="", text="x", markdown="x",
              embedding_text="x", pages=[1]),
    ])
    registry.save_split(DOC_ID, SplitResult(sections=[section]))
    got = registry.get_document(DOC_ID)
    assert len(got["chunks"]) == 1
    assert got["document"]["chunk_count"] == 1
    assert got["document"]["category"] == "invoice"   # untouched by save_split


def test_updated_at_bumps_on_mutation():
    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.storage import registry

    _save_all()
    first = registry.get_document(DOC_ID)["document"]["updated_at"]
    assert first is not None                       # set on insert (server_default now())

    # a real mutation (re-classify to a new category) must bump updated_at
    registry.save_classify(DOC_ID, ClassifyResult(category="receipt", confidence=0.5, reasoning="changed"))
    second = registry.get_document(DOC_ID)["document"]["updated_at"]
    assert second > first                          # bumped by onupdate on the UPDATE


def test_delete_cascades():
    from ingestlib.storage import registry

    _save_all()
    assert registry.document_exists(DOC_ID) is True
    assert registry.delete_document(DOC_ID) is True
    assert registry.document_exists(DOC_ID) is False
    assert registry.get_document(DOC_ID) is None
    assert registry.delete_document(DOC_ID) is False   # already gone
