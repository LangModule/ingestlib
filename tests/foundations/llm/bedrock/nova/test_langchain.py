"""LangChain path via ChatBedrockConverse — for callers who want chain composition."""
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from ingestlib.foundations.llm import get_llm, get_llm_with_thinking


def test_get_llm_returns_ChatBedrockConverse():
    assert isinstance(get_llm(), ChatBedrockConverse)


def test_get_llm_invoke_returns_string_content():
    llm = get_llm()
    resp = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
    assert isinstance(resp.content, str)
    assert "OK" in resp.content


def test_get_llm_singleton_cache_by_params():
    a = get_llm(max_tokens=8192, temperature=0.0)
    b = get_llm(max_tokens=8192, temperature=0.0)
    assert a is b


def test_get_llm_different_max_tokens_different_instances():
    a = get_llm(max_tokens=8192)
    b = get_llm(max_tokens=16384)
    assert a is not b


def test_get_llm_with_thinking_returns_ChatBedrockConverse():
    assert isinstance(get_llm_with_thinking(), ChatBedrockConverse)


def test_get_llm_with_thinking_invoke_returns_content_blocks():
    llm = get_llm_with_thinking(effort="low")
    resp = llm.invoke([HumanMessage(content="What is 4+4? Just the number.")])
    assert isinstance(resp.content, list)
    text_blocks = [b for b in resp.content if isinstance(b, dict) and b.get("type") == "text"]
    assert text_blocks, f"expected a text block in {resp.content!r}"
    assert "8" in text_blocks[0]["text"]


def test_get_llm_with_thinking_cache_by_effort():
    a = get_llm_with_thinking(effort="low")
    b = get_llm_with_thinking(effort="low")
    c = get_llm_with_thinking(effort="medium")
    assert a is b
    assert a is not c
