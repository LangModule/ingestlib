"""Real verification of chat_structured() — schema-enforced output via tool-forcing."""
import pytest
from pydantic import BaseModel, Field

from ingestlib.foundations.llm import Image, achat_structured, chat_structured


class _Verdict(BaseModel):
    category: str = Field(description="snake_case label")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


_INVOICE_TEXT = (
    "Classify this document. Content: INVOICE #4821 from Acme Corp. "
    "12 widgets @ $8 each. Total due: $96. Payment terms: Net 30."
)


def test_returns_validated_schema_instance():
    v = chat_structured(_INVOICE_TEXT, schema=_Verdict)
    assert isinstance(v, _Verdict)
    assert v.category == "invoice"
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasoning.strip()


def test_system_prompt_accepted_without_error():
    """System prompt is passed through; with tool-forcing the schema still wins,
    so assert the mechanism works rather than model obedience to style rules."""
    v = chat_structured(
        _INVOICE_TEXT,
        schema=_Verdict,
        system="You are a strict document classification engine.",
    )
    assert isinstance(v, _Verdict)
    assert v.category.strip()


def test_with_image_input(doc_chart_bytes):
    class _ImageVerdict(BaseModel):
        is_document_page: bool
        summary: str

    v = chat_structured(
        "Is this image a document page (vs a photograph)? Summarize it in one sentence.",
        schema=_ImageVerdict,
        images=[Image(doc_chart_bytes, "png")],
    )
    assert isinstance(v, _ImageVerdict)
    assert v.is_document_page is True


async def test_achat_structured_matches_sync_shape():
    v = await achat_structured(_INVOICE_TEXT, schema=_Verdict)
    assert isinstance(v, _Verdict)
    assert v.category == "invoice"


def test_nested_schema_with_list():
    class _Item(BaseModel):
        name: str
        quantity: int

    class _Extraction(BaseModel):
        vendor: str
        items: list[_Item]
        total: float

    v = chat_structured(
        "Extract the order: INVOICE from Acme Corp. 12 widgets, 3 gadgets. Total $96.50.",
        schema=_Extraction,
    )
    assert "acme" in v.vendor.lower()
    assert len(v.items) == 2
    assert sorted(i.quantity for i in v.items) == [3, 12]
    assert v.total == pytest.approx(96.50)


def test_invalid_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        chat_structured("hi", schema=_Verdict, max_tokens=1024)  # type: ignore[arg-type]
