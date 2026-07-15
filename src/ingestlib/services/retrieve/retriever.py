"""retrieve() / aretrieve() — question in, ranked cited chunks out.

Cascading retrieval: dense vector search plus, on hybrid stores, lexical
sparse search over the same chunks — then Jina reranking on the merged
candidates (the reranker reads full text, so it both catches what embedding
similarity misses AND produces one comparable order from the two incomparable
score scales). Every hit carries provenance — document, pages, region_ids —
so answers can cite their exact source location.
"""
import asyncio
from typing import Any

from ingestlib.foundations.llm import aembed_text, jina_arerank
from ingestlib.services.retrieve.models import Hit, RetrievalResult
from ingestlib.storage import VectorStore, default_store
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# With reranking on, fetch a wider candidate pool for the reranker to sort.
_CANDIDATE_MULTIPLIER = 4


async def aretrieve(
    question: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    namespace: str = "",
    rerank: bool = True,
    store: VectorStore | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant chunks for a question (async).

    question — natural-language query
    top_k    — hits to return
    filters  — payload constraints, e.g. {"category": "research_paper"}
    rerank   — rerank candidates with Jina (recommended; needs JINA_API_KEY)
    store    — vector store connector; defaults to the one selected by
               config.yaml's `vector_store` key
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")
    store = store or default_store()

    vector = await aembed_text(question, purpose="GENERIC_RETRIEVAL")
    # store.query is a sync SDK network call — keep it off the event loop
    candidates = await asyncio.to_thread(
        store.query,
        vector,
        top_k=top_k * _CANDIDATE_MULTIPLIER if rerank else top_k,
        filters=filters,
        namespace=namespace,
        text=question,  # hybrid stores add lexical hits; dense-only stores ignore it
    )
    if not candidates:
        logger.info("retrieve: no hits for %r", question[:60])
        return RetrievalResult(question=question)

    if not rerank or len(candidates) == 1:
        hits = [Hit(chunk=c, vector_score=c.score) for c in candidates[:top_k]]
        return RetrievalResult(question=question, hits=hits)

    documents = [c.markdown or c.text for c in candidates]
    try:
        ranking = await jina_arerank(question, documents, top_n=top_k)
    except Exception as exc:
        # retrieval must not die because the reranker hiccuped — degrade to
        # vector order and say so loudly
        logger.warning("rerank failed (%s: %s) — returning vector order", type(exc).__name__, exc)
        hits = [Hit(chunk=c, vector_score=c.score) for c in candidates[:top_k]]
        return RetrievalResult(question=question, hits=hits)
    hits = [
        Hit(chunk=candidates[idx], vector_score=candidates[idx].score, rerank_score=score)
        for idx, score in ranking
    ]
    logger.info(
        "retrieve: %d candidate(s) → %d reranked hit(s) for %r",
        len(candidates), len(hits), question[:60],
    )
    return RetrievalResult(question=question, hits=hits)


def retrieve(
    question: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    namespace: str = "",
    rerank: bool = True,
    store: VectorStore | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant chunks for a question. Sync wrapper — use
    aretrieve() inside an event loop."""
    return asyncio.run(aretrieve(
        question, top_k=top_k, filters=filters,
        namespace=namespace, rerank=rerank, store=store,
    ))
