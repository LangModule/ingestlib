"""Async wrappers produce identical results to sync at temp=0."""
import asyncio

from ingestlib.foundations.llm import achat, achat_with_thinking, chat


async def test_achat_matches_sync():
    prompt = "Reply with exactly: OK"
    sync = chat(prompt)
    async_ = await achat(prompt)
    assert sync == async_


async def test_achat_with_thinking_returns_correct_answer():
    r = await achat_with_thinking("What is 3+3? Just the number.", effort="low", max_tokens=32768)
    assert isinstance(r, str)
    assert "6" in r


async def test_gather_runs_concurrent_calls():
    results = await asyncio.gather(
        achat("Reply with exactly: ONE"),
        achat("Reply with exactly: TWO"),
        achat("Reply with exactly: THREE"),
    )
    assert len(results) == 3
    assert all(isinstance(r, str) and r for r in results)
    assert "ONE" in results[0].upper()
    assert "TWO" in results[1].upper()
    assert "THREE" in results[2].upper()
