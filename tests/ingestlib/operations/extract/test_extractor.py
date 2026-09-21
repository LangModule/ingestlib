"""The extractor's LLM-free machinery — response schemas, merge, dedup. Pure."""
import pytest
from pydantic import BaseModel

from ingestlib.operations.extract import ExtractedItem, FieldValue, extract
from ingestlib.operations.extract.extractor import (
    _dedup_many,
    _merge_one,
    _response_models,
)


class _Receipt(BaseModel):
    merchant: str
    total: float


def test_invalid_mode_raises_before_touching_the_source():
    with pytest.raises(ValueError, match="mode"):
        extract("does-not-exist.pdf", schema=_Receipt, mode="both")


def test_response_models_wrap_the_caller_schema():
    item_model, many_model = _response_models(_Receipt)
    raw = item_model.model_validate({
        "data": {"merchant": "BART", "total": 20.0},
        "evidence": [{"field": "total", "sources": ["p1:r2"]}],
    })
    assert raw.data.total == 20.0
    # confidence is OPTIONAL with a baseline — models frequently omit
    # self-scores, and that must never fail the extraction
    assert raw.evidence[0].confidence == 0.7

    batch = many_model.model_validate({"items": []})
    assert batch.items == []


def _item(merchant: str, conf_merchant: float, conf_total: float) -> ExtractedItem:
    return ExtractedItem(
        value=_Receipt(merchant=merchant, total=20.0),
        fields={
            "merchant": FieldValue(confidence=conf_merchant, pages=[1]),
            "total": FieldValue(confidence=conf_total, pages=[2]),
        },
        pages=[1, 2],
    )


def test_merge_one_picks_highest_confidence_per_field():
    a = _item("From Window A", conf_merchant=0.9, conf_total=0.2)
    b = _item("From Window B", conf_merchant=0.4, conf_total=0.8)
    merged = _merge_one([a, b], _Receipt)
    assert merged.value.merchant == "From Window A"     # a wins merchant
    assert merged.fields["total"].confidence == 0.8     # b wins total
    assert isinstance(merged.value, _Receipt), "merged data revalidates"


def test_merge_one_single_item_passes_through():
    a = _item("Only", 0.9, 0.9)
    assert _merge_one([a], _Receipt) is a


def test_dedup_many_collapses_identical_values_keeps_distinct():
    a = _item("BART", 0.9, 0.9)
    duplicate = _item("BART", 0.5, 0.5)          # same VALUE, different provenance
    distinct = ExtractedItem(
        value=_Receipt(merchant="Safeway", total=20.0),
        fields={"merchant": FieldValue(confidence=0.9),
                "total": FieldValue(confidence=0.9)},
        pages=[14],
    )
    unique = _dedup_many([a, duplicate, distinct])
    assert len(unique) == 2
    assert unique[0] is a, "first sighting wins"
    assert unique[1].value.merchant == "Safeway"
