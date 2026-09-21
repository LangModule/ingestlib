"""Real verification of chat() — text-only and multimodal against Nova 2 Lite."""
import pytest

from ingestlib.foundations.llm import Image


def test_chat_system_prompt_changes_output(llm):
    without = llm.chat("What is 2+2? Answer just the number.")
    with_sys = llm.chat(
        "What is 2+2? Answer just the number.",
        system="You must reply only in French. Never use English or digits.",
    )
    assert without != with_sys, "system prompt should influence output"


def test_chat_with_single_image_identifies_content(llm, photo_bytes):
    r = llm.chat(
        "What animal is in this image? Reply with one lowercase word only.",
        images=[Image(photo_bytes, "jpeg")],
    )
    assert "cat" in r.lower(), f"expected 'cat' in response, got {r!r}"


def test_chat_with_multiple_images_reads_both(llm, doc_text_bytes, doc_chart_bytes):
    r = llm.chat(
        "Are these two images photos of animals or scanned documents? Answer with one word.",
        images=[Image(doc_text_bytes, "png"), Image(doc_chart_bytes, "png")],
    )
    assert "document" in r.lower(), f"expected 'document' in response, got {r!r}"


@pytest.mark.parametrize("max_tokens", [8192, 16384, 32768, 65535])
def test_chat_max_tokens_variants_all_accepted(llm, max_tokens):
    r = llm.chat("Say hi in one word.", max_tokens=max_tokens)
    assert isinstance(r, str) and r.strip()
