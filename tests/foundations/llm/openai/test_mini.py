"""Real verification of GPT-5 chat — text, vision, thinking, structured.

Skipped when OPENAI_API_KEY is not set (CI without secrets, fresh clones).
"""
import os

import pytest
from pydantic import BaseModel, Field

from ingestlib.foundations.llm import Image

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set in .env",
)


def test_chat_returns_text(oai):
    r = oai.chat("What is 2+2? Reply with just the number.")
    assert isinstance(r, str)
    assert "4" in r


def test_chat_system_prompt_changes_output(oai):
    without = oai.chat("What is 2+2? Answer just the number.")
    with_sys = oai.chat(
        "What is 2+2? Answer just the number.",
        system="You must reply only in French words. Never use digits.",
    )
    assert without != with_sys, "system prompt should influence output"


def test_chat_with_image_identifies_content(oai, photo_bytes):
    r = oai.chat(
        "What animal is in this image? Reply with one lowercase word only.",
        images=[Image(photo_bytes, "jpeg")],
    )
    assert "cat" in r.lower(), f"expected 'cat' in response, got {r!r}"


def test_chat_reads_document_page(oai, doc_chart_bytes):
    r = oai.chat(
        "Is this image a document page or a photograph of an animal? One word.",
        images=[Image(doc_chart_bytes, "png")],
    )
    assert "document" in r.lower(), f"expected 'document' in response, got {r!r}"


def test_thinking_answers_correctly():
    from ingestlib.foundations.llm.openai import chat_with_thinking

    r = chat_with_thinking("What is 17 + 25? Reply with only the number.", effort="low")
    assert "42" in r


class _Verdict(BaseModel):
    category: str = Field(description="snake_case label")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def test_structured_returns_validated_schema_instance(oai):
    v = oai.structured(
        "Classify this document. Content: INVOICE #4821 from Acme Corp. "
        "12 widgets @ $8 each. Total due: $96. Payment terms: Net 30.",
        _Verdict,
    )
    assert isinstance(v, _Verdict)
    assert v.category == "invoice"
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasoning.strip()


def test_structured_nested_schema_with_list(oai):
    class _Item(BaseModel):
        name: str
        quantity: int

    class _Extraction(BaseModel):
        vendor: str
        items: list[_Item]
        total: float

    v = oai.structured(
        "Extract the order: INVOICE from Acme Corp. 12 widgets, 3 gadgets. Total $96.50.",
        _Extraction,
    )
    assert "acme" in v.vendor.lower()
    assert sorted(i.quantity for i in v.items) == [3, 12]
    assert v.total == pytest.approx(96.50)


async def test_achat_structured_matches_sync_shape():
    from ingestlib.foundations.llm.openai import achat_structured

    v = await achat_structured(
        "Classify this document. Content: INVOICE #4821 from Acme Corp, total $96.",
        _Verdict,
    )
    assert isinstance(v, _Verdict)
    assert v.category == "invoice"


async def test_achat_answers_correctly():
    from ingestlib.foundations.llm.openai import achat

    r = await achat("What is 3+3? Reply with just the number.")
    assert isinstance(r, str)
    assert "6" in r


# ---------- LangChain surface + caching (builds clients, no chat calls) ----------


def test_get_llm_returns_ChatOpenAI():
    from langchain_openai import ChatOpenAI

    from ingestlib.foundations.llm.openai import get_llm

    assert isinstance(get_llm(), ChatOpenAI)


def test_get_llm_singleton_cache_by_params():
    from ingestlib.foundations.llm.openai import get_llm

    a = get_llm(max_tokens=8192)
    b = get_llm(max_tokens=8192)
    c = get_llm(max_tokens=16384)
    assert a is b
    assert a is not c


def test_get_llm_with_thinking_cache_by_effort():
    from ingestlib.foundations.llm.openai import get_llm, get_llm_with_thinking

    a = get_llm_with_thinking(effort="low")
    b = get_llm_with_thinking(effort="low")
    c = get_llm_with_thinking(effort="medium")
    assert a is b
    assert a is not c
    assert get_llm(max_tokens=32768) is not a, "minimal-effort chat must not share thinking instances"


def test_reset_models_drops_cached_instances():
    from ingestlib.foundations.llm.openai import get_llm, reset_models

    before = get_llm()
    reset_models()
    assert get_llm() is not before
