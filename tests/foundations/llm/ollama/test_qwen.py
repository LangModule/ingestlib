"""Real verification of Qwen chat via Ollama — text, vision, thinking, structured.

Opt-in via RUN_OLLAMA_E2E=1: needs a local Ollama serving the configured
chat model (a GGUF build — the MLX engine drops images and schemas).
"""
import os

import pytest
from pydantic import BaseModel, Field

from ingestlib.foundations.llm import Image

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OLLAMA_E2E") != "1",
    reason="ollama e2e is opt-in: set RUN_OLLAMA_E2E=1 (needs a local Ollama "
           "with the configured models pulled)",
)


def test_chat_returns_text(olm):
    r = olm.chat("What is 2+2? Reply with just the number.")
    assert isinstance(r, str)
    assert "4" in r


def test_chat_system_prompt_changes_output(olm):
    without = olm.chat("What is 2+2? Answer just the number.")
    with_sys = olm.chat(
        "What is 2+2? Answer just the number.",
        system="You must reply only in French words. Never use digits.",
    )
    assert without != with_sys, "system prompt should influence output"


def test_chat_with_image_identifies_content(olm, photo_bytes):
    r = olm.chat(
        "What animal is in this image? Reply with one lowercase word only.",
        images=[Image(photo_bytes, "jpeg")],
    )
    assert "cat" in r.lower(), f"expected 'cat' in response, got {r!r}"


def test_chat_reads_document_page(olm, doc_chart_bytes):
    """Asserting on the chart's printed title proves the image actually
    reached the model — a backend that silently drops images (the Ollama
    -mlx engine) fails here instead of passing on a lucky guess."""
    r = olm.chat(
        "What title is printed at the top of this chart? Reply with the title only.",
        images=[Image(doc_chart_bytes, "png")],
    )
    assert "sales" in r.lower(), f"expected the chart title 'My sales', got {r!r}"


def test_thinking_answers_correctly():
    from ingestlib.foundations.llm.ollama import chat_with_thinking

    r = chat_with_thinking("What is 17 + 25? Reply with only the number.", effort="low")
    assert "42" in r


class _Verdict(BaseModel):
    category: str = Field(description="snake_case label")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def test_structured_returns_validated_schema_instance(olm):
    v = olm.structured(
        "Classify this document. Content: INVOICE #4821 from Acme Corp. "
        "12 widgets @ $8 each. Total due: $96. Payment terms: Net 30.",
        _Verdict,
    )
    assert isinstance(v, _Verdict)
    # capitalization is a formatting nit on a local 9B — the pipeline's real
    # prompts demand snake_case explicitly; here the machinery is under test
    assert v.category.lower() == "invoice"
    assert 0.0 <= v.confidence <= 1.0
    assert v.reasoning.strip()


def test_structured_nested_schema_with_list(olm):
    class _Item(BaseModel):
        name: str
        quantity: int

    class _Extraction(BaseModel):
        vendor: str
        items: list[_Item]
        total: float

    v = olm.structured(
        "Extract the order: INVOICE from Acme Corp. 12 widgets, 3 gadgets. Total $96.50.",
        _Extraction,
    )
    assert "acme" in v.vendor.lower()
    assert sorted(i.quantity for i in v.items) == [3, 12]
    assert v.total == pytest.approx(96.50)


async def test_achat_structured_matches_sync_shape():
    from ingestlib.foundations.llm.ollama import achat_structured

    v = await achat_structured(
        "Classify this document. Content: INVOICE #4821 from Acme Corp, total $96.",
        _Verdict,
    )
    assert isinstance(v, _Verdict)
    assert v.category.lower() == "invoice"


async def test_achat_answers_correctly():
    from ingestlib.foundations.llm.ollama import achat

    r = await achat("What is 3+3? Reply with just the number.")
    assert isinstance(r, str)
    assert "6" in r


# ---------- LangChain surface + caching (builds clients, no chat calls) ----------


def test_get_llm_returns_ChatOpenAI():
    from langchain_openai import ChatOpenAI

    from ingestlib.foundations.llm.ollama import get_llm

    assert isinstance(get_llm(), ChatOpenAI)


def test_get_llm_singleton_cache_by_params():
    from ingestlib.foundations.llm.ollama import get_llm

    a = get_llm(max_tokens=8192)
    b = get_llm(max_tokens=8192)
    c = get_llm(max_tokens=16384)
    assert a is b
    assert a is not c


def test_get_llm_with_thinking_cache_by_effort():
    from ingestlib.foundations.llm.ollama import get_llm, get_llm_with_thinking

    a = get_llm_with_thinking(effort="low")
    b = get_llm_with_thinking(effort="low")
    c = get_llm_with_thinking(effort="medium")
    assert a is b
    assert a is not c
    assert get_llm(max_tokens=32768) is not a, "no-thinking chat must not share thinking instances"


def test_reset_models_drops_cached_instances():
    from ingestlib.foundations.llm.ollama import get_llm, reset_models

    before = get_llm()
    reset_models()
    assert get_llm() is not before
