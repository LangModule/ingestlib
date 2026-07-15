"""Amazon Rerank v1 primitives (sync and async) via bedrock-agent-runtime."""
import asyncio
import time
from typing import Any

from ingestlib.config import get_bedrock_config
from ingestlib.foundations.llm.bedrock.factory import get_rerank_agent_client
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)
MAX_DOCUMENTS: int = 1000


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
    if len(documents) > MAX_DOCUMENTS:
        raise ValueError(
            f"documents must contain at most {MAX_DOCUMENTS} items, got {len(documents)}"
        )

    client = get_rerank_agent_client()
    cfg = get_bedrock_config()
    model_arn = f"arn:aws:bedrock:{cfg.rerank_region}::foundation-model/{cfg.rerank_model_id}"

    reranking_config: dict[str, Any] = {"modelConfiguration": {"modelArn": model_arn}}
    if top_n is not None:
        reranking_config["numberOfResults"] = top_n

    logger.info(
        "AWS rerank: model=%s query_len=%d n_docs=%d top_n=%s",
        cfg.rerank_model_id, len(query), len(documents), top_n,
    )
    t0 = time.perf_counter()
    request: dict[str, Any] = {
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": reranking_config,
        },
        "sources": [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": doc},
                },
            }
            for doc in documents
        ],
        "queries": [{"type": "TEXT", "textQuery": {"text": query}}],
    }
    # the Rerank API paginates — follow nextToken or large calls silently truncate
    results: list[tuple[int, float]] = []
    next_token: str | None = None
    while True:
        response = client.rerank(**request, **({"nextToken": next_token} if next_token else {}))
        results.extend((r["index"], r["relevanceScore"]) for r in response["results"])
        next_token = response.get("nextToken")
        if not next_token:
            break
    logger.info(
        "AWS rerank done: %.2fs returned=%d",
        time.perf_counter() - t0, len(results),
    )
    return results


async def arerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Async rerank() — runs the sync Bedrock client in a worker thread."""
    return await asyncio.to_thread(rerank, query, documents, top_n)
