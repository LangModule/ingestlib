"""Client-side validation guards."""
import pytest

from ingestlib.foundations.llm import chat, chat_with_thinking, get_llm, get_llm_with_thinking


def test_chat_invalid_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        chat("hi", max_tokens=1024)


def test_chat_with_thinking_invalid_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        chat_with_thinking("hi", max_tokens=1024)


def test_get_llm_invalid_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        get_llm(max_tokens=1024)


def test_get_llm_with_thinking_invalid_max_tokens_raises():
    with pytest.raises(ValueError, match="max_tokens"):
        get_llm_with_thinking(max_tokens=1024)
