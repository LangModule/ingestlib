"""Extract result models — pure, always run."""
import pytest
from pydantic import BaseModel

from ingestlib.operations.extract import ExtractedItem, ExtractResult, FieldValue


class _Receipt(BaseModel):
    merchant: str
    total: float


def _item() -> ExtractedItem:
    return ExtractedItem(
        value=_Receipt(merchant="BART", total=20.0),
        fields={
            "merchant": FieldValue(confidence=0.9, region_ids={10: [1]}, pages=[10],
                                   grounded=True),
            "total": FieldValue(confidence=0.95, region_ids={10: [3]}, pages=[10],
                                grounded=True),
        },
        pages=[10],
    )


def test_models_are_frozen():
    item = _item()
    with pytest.raises(Exception):
        item.pages = [1]  # type: ignore[misc]
    with pytest.raises(Exception):
        item.fields["total"].confidence = 0.1  # type: ignore[misc]


def test_citation_formats_pages():
    assert _item().citation == "p.10"
    multi = _item().model_copy(update={"pages": [1, 2]})
    assert multi.citation == "p.1,2"
    none = _item().model_copy(update={"pages": []})
    assert none.citation == "p.?"


def test_values_property_returns_schema_instances_in_order():
    result = ExtractResult(
        items=[_item(), _item()], schema_name="_Receipt", mode="many",
    )
    assert [v.merchant for v in result.values] == ["BART", "BART"]
    assert all(isinstance(v, _Receipt) for v in result.values)


def test_serialization_round_trip_dumps_value_to_dict():
    """The value is the CALLER's model — it must serialize as plain data and
    revalidate back (load_extract's contract)."""
    result = ExtractResult(items=[_item()], schema_name="_Receipt", mode="one")
    payload = result.model_dump(mode="json")
    assert payload["items"][0]["value"] == {"merchant": "BART", "total": 20.0}

    restored = ExtractResult.model_validate(payload)
    assert restored.items[0].value == {"merchant": "BART", "total": 20.0}  # dict until revalidated
    revalidated = _Receipt.model_validate(restored.items[0].value)
    assert revalidated.total == 20.0
    assert restored.items[0].fields["total"].region_ids == {10: [3]}


def test_field_value_confidence_bounds():
    with pytest.raises(Exception):
        FieldValue(confidence=1.5)
    with pytest.raises(Exception):
        FieldValue(confidence=-0.1)
