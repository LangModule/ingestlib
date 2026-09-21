"""recollect() — re-classify stored docs from the registry (no OCR) and re-sort.

Stubs the classify OPERATION so no LLM is needed; checks the service re-labels
category + collection in the registry and reports what moved.
"""


def test_recollect_resorts_from_the_registry(pipeline, monkeypatch):
    import importlib

    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.services import get_document, ingest, recollect
    from ingestlib.utils.files import sha256_of_file

    source = pipeline.corpus / "doc.pdf"
    source.write_bytes(b"%PDF recollect fixture")
    ingest(source, store=pipeline.store)  # pipeline stub → category "report"
    doc_id = sha256_of_file(source)
    assert get_document(doc_id).collection == "report"

    # rules changed: recollect re-classifies to "invoice" (stubbed, no LLM)
    op_classify = importlib.import_module("ingestlib.operations.classify")

    async def fake_classify(src, categories=None, *, target_pages=None, max_pages=None):
        # the reconstructed, text-only ParseResult reaches here
        assert src.pages[0].text == "hello"
        return ClassifyResult(category="invoice", confidence=0.8)

    monkeypatch.setattr(op_classify, "aclassify", fake_classify)

    result = recollect()
    assert result.recollected >= 1
    assert any(c.doc_id == doc_id and c.from_category == "report"
               and c.to_category == "invoice" for c in result.changed)

    doc = get_document(doc_id)
    assert doc.category == "invoice"
    assert doc.collection == "invoice"


def test_recollect_materializes_declared_collections(pipeline, monkeypatch):
    """A rules change that adds a collection's schema/auto_extract is applied by
    recollect — not just the per-document label, but the collections table too."""
    import dataclasses

    import ingestlib.config as config_module
    from ingestlib.config import CollectionRule, get_config
    from ingestlib.services import recollect
    from ingestlib.storage import registry

    schema = {"type": "object", "properties": {"total": {"type": "number"}}}
    cfg = get_config()
    classify = dataclasses.replace(
        cfg.classify,
        collections={"report": CollectionRule(
            description="quarterly report", extract_schema=schema, auto_extract=True
        )},
    )
    monkeypatch.setattr(config_module, "_config", dataclasses.replace(cfg, classify=classify))

    recollect()

    coll = registry.get_collection("report")
    assert coll is not None
    assert coll["description"] == "quarterly report"
    assert coll["extract_schema"] == schema
    assert coll["auto_extract"] is True
