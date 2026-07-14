"""Jina AI Reranker — POST {base_url}/rerank, Bearer auth via JINA_API_KEY."""
import asyncio
import time
from typing import Any

import httpx

from ingestlib.config import get_jina_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Attempts per call — 429s back off and retry before giving up.
_MAX_ATTEMPTS = 3


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Rerank documents against a query.

    Returns (original_index, relevance_score) pairs sorted by score descending.
    """
    if not documents:
        raise ValueError("documents must contain at least one item")

    cfg = get_jina_config()
    if not cfg.api_key:
        raise RuntimeError("JINA_API_KEY is not set — add it to .env")

    payload: dict[str, Any] = {
        "model": cfg.rerank_model_id,
        "query": query,
        "documents": documents,
        "return_documents": False,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    logger.info(
        "Jina rerank: model=%s query_len=%d n_docs=%d top_n=%s",
        cfg.rerank_model_id, len(query), len(documents), top_n,
    )
    t0 = time.perf_counter()
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        response = httpx.post(
            f"{cfg.base_url}/rerank",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            json=payload,
            timeout=60.0,
        )
        if response.status_code == 429 and attempt < _MAX_ATTEMPTS:
            # rate-limited — honor Retry-After when present, else back off
            wait = float(response.headers.get("retry-after", 2.0 * attempt))
            logger.warning(
                "Jina rerank rate-limited (429) — retrying in %.0fs (attempt %d/%d)",
                wait, attempt, _MAX_ATTEMPTS,
            )
            time.sleep(wait)
            continue
        break
    response.raise_for_status()
    results = [(r["index"], r["relevance_score"]) for r in response.json()["results"]]
    logger.info(
        "Jina rerank done: %.2fs returned=%d",
        time.perf_counter() - t0, len(results),
    )
    return results


async def arerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Async rerank() — runs the sync HTTP call in a worker thread."""
    return await asyncio.to_thread(rerank, query, documents, top_n)
