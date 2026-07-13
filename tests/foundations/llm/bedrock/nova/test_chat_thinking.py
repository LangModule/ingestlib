"""Real verification of chat_with_thinking() — reasoning mode with all three effort levels."""
import pytest

from ingestlib.foundations.llm import Image


def test_thinking_medium_returns_correct_answer(llm):
    r = llm.chat_with_thinking("What is 2+2? Reply with just the number.")
    assert isinstance(r, str)
    assert "4" in r


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_all_effort_levels_produce_correct_answer(llm, effort):
    r = llm.chat_with_thinking(
        "What is 17 + 25? Reply with only the number, no other text.",
        effort=effort,
        max_tokens=32768,
    )
    assert "42" in r, f"effort={effort} gave wrong answer: {r!r}"


def test_thinking_high_effort_does_not_error(llm):
    """high effort requires temperature to be omitted server-side; our code handles this."""
    r = llm.chat_with_thinking(
        "What is 7+8? Just the number.",
        effort="high",
        max_tokens=32768,
    )
    assert isinstance(r, str)
    assert "15" in r


def test_thinking_multimodal(llm, photo_bytes):
    r = llm.chat_with_thinking(
        "What animal is in this image? One lowercase word.",
        images=[Image(photo_bytes, "jpeg")],
        effort="low",
    )
    assert "cat" in r.lower()
