"""The extract service persists to the registry (extract is no longer an island).

Stubs the extract OPERATION (no LLM) and checks the SERVICE's persist wiring:
save to the extractions table → retrievable via get_document.
"""


def test_extract_persists_and_is_retrievable(pipeline, monkeypatch):
    from pydantic import BaseModel

    import importlib

    op_extract = importlib.import_module("ingestlib.operations.extract")
    from ingestlib.operations.extract.models import ExtractedItem, ExtractResult, FieldValue
    from ingestlib.services import extract, get_document, ingest
    from ingestlib.utils.files import sha256_of_file

    source = pipeline.corpus / "invoice.pdf"
    source.write_bytes(b"%PDF extract persist fixture")
    ingest(source, store=pipeline.store)
    doc_id = sha256_of_file(source)

    class Invoice(BaseModel):
        total: float

    async def fake_op_extract(src, schema, *, mode="one", target_pages=None, instructions=None):
        return ExtractResult(
            items=[ExtractedItem(
                value=Invoice(total=42.0),
                fields={"total": FieldValue(confidence=0.9, pages=[1], grounded=True)},
                pages=[1],
            )],
            schema_name="Invoice", mode="one",
        )

    monkeypatch.setattr(op_extract, "aextract", fake_op_extract)

    result = extract(source, Invoice, persist=True)
    assert result.items[0].value.total == 42.0

    doc = get_document(doc_id)
    assert len(doc.extractions) == 1
    assert doc.extractions[0]["schema_name"] == "Invoice"
    assert doc.extractions[0]["value"] == {"total": 42.0}
    assert doc.extractions[0]["fields"]["total"]["grounded"] is True


def test_persist_requires_the_document_in_the_corpus(pipeline, monkeypatch):
    import pytest
    from pydantic import BaseModel

    import importlib

    op_extract = importlib.import_module("ingestlib.operations.extract")
    from ingestlib.operations.extract.models import ExtractResult
    from ingestlib.services import extract

    source = pipeline.corpus / "stranger.pdf"
    source.write_bytes(b"never ingested")

    class Thing(BaseModel):
        x: int

    async def fake_op_extract(src, schema, *, mode="one", target_pages=None, instructions=None):
        return ExtractResult(items=[], schema_name="Thing", mode="one")

    monkeypatch.setattr(op_extract, "aextract", fake_op_extract)

    with pytest.raises(ValueError, match="not in the corpus"):
        extract(source, Thing, persist=True)
