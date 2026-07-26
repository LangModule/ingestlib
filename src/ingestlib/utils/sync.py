"""Bridge for the sync wrappers — asyncio.run with an honest event-loop error.

Inside Jupyter (or any code already running an event loop) asyncio.run()
raises a RuntimeError that never mentions the fix. Every sync wrapper runs
through here so the error names the async form to await instead.
"""
import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T], async_name: str) -> T:
    """asyncio.run(coro), naming `async_name` when a loop is already running."""
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "running event loop" in str(exc):
            coro.close()  # never awaited — close it quietly
            raise RuntimeError(
                f"you are inside a running event loop (a notebook?) — call "
                f"`await {async_name}(...)` instead of the sync form"
            ) from None
        raise
