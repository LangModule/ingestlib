"""retrieve() / aretrieve() — question in, ranked cited results out.

Documents (the default): dense vector search plus, on hybrid stores, lexical
sparse search over the same chunks — then reranking on the merged candidates
(the reranker reads full text, so it both catches what embedding similarity
misses AND produces one comparable order from the two incomparable score
scales). The reranker is selected by config.yaml's `reranker` key.

With `sources=[...]`, retrieve fans out over declared sources (the document
corpus AND/OR SQL databases from sources.yaml) and merges their normalized
SourceResults into one envelope — the caller never picks a backend.
"""
import asyncio
from typing import Any

from ingestlib.config import get_config
from ingestlib.foundations.llm import aembed_text, jina_arerank
from ingestlib.services.retrieve.models import Hit, RetrievalResult
from ingestlib.storage import VectorStore, default_store
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)

# With reranking on, fetch a wider candidate pool for the reranker to sort.
_CANDIDATE_MULTIPLIER = 4

# config.yaml `reranker` key → implementation ("none" short-circuits instead).
# The aws entry resolves lazily so a jina/none pipeline never imports bedrock.
_RERANKER_NAMES = ("jina", "aws")


def _reranker(name: str):
    if name == "jina":
        return jina_arerank
    from ingestlib.foundations.llm import aws_arerank

    return aws_arerank


async def aretrieve(
    question: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    namespace: str = "",
    rerank: bool = True,
    store: VectorStore | None = None,
    sources: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant results for a question (async).

    question — natural-language query
    top_k    — results to return per source
    filters  — payload constraints for document search, e.g. {"category": "x"}
    rerank   — rerank document candidates with config.yaml's `reranker`
    store    — vector store connector; defaults to config.yaml's `vector_store`
    sources  — names from sources.yaml to query (documents and/or SQL databases).
               When given, retrieve fans out over them and returns a normalized
               envelope (result.results); omit it for plain document search
               (result.hits) — the exact prior behavior.
    """
    if not question.strip():
        raise ValueError("question must be a non-empty string")

    if sources:
        from ingestlib.sources.registry import resolve_sources

        resolved = resolve_sources(sources)
        gathered = await asyncio.gather(*(s.answer(question, top_k=top_k) for s in resolved))
        results = [r for group in gathered for r in group]
        logger.info("retrieve: %d source(s) → %d result(s) for %r",
                    len(resolved), len(results), question[:60])
        return RetrievalResult(question=question, results=results)

    hits = await _retrieve_document_hits(
        question, top_k=top_k, filters=filters, namespace=namespace,
        rerank=rerank, store=store,
    )
    return RetrievalResult(question=question, hits=hits)


async def _retrieve_document_hits(
    question: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    namespace: str = "",
    rerank: bool = True,
    store: VectorStore | None = None,
) -> list[Hit]:
    """The dense + rerank document retrieval — returns ranked Hits.

    Shared by aretrieve() (the default path) and the DocumentSource wrapper, so
    the two never diverge.
    """
    store = store or default_store()

    reranker = get_config().reranker
    if reranker != "none" and reranker not in _RERANKER_NAMES:
        raise ValueError(
            f"unknown reranker {reranker!r} in config.yaml — "
            f"choose one of {sorted(_RERANKER_NAMES) + ['none']}"
        )
    use_rerank = rerank and reranker != "none"

    vector = await aembed_text(question, purpose="GENERIC_RETRIEVAL")
    # store.query is a sync SDK network call — keep it off the event loop
    candidates = await asyncio.to_thread(
        store.query,
        vector,
        top_k=top_k * _CANDIDATE_MULTIPLIER if use_rerank else top_k,
        filters=filters,
        namespace=namespace,
        text=question,  # hybrid stores add lexical hits; dense-only stores ignore it
    )
    if not candidates:
        logger.info("retrieve: no hits for %r", question[:60])
        return []

    if not use_rerank or len(candidates) == 1:
        return [Hit(chunk=c, vector_score=c.score) for c in candidates[:top_k]]

    documents = [c.markdown or c.text for c in candidates]
    try:
        ranking = await _reranker(reranker)(question, documents, top_n=top_k)
    except Exception as exc:
        # retrieval must not die because the reranker hiccuped — degrade to
        # vector order and say so loudly
        logger.warning("rerank failed (%s: %s) — returning vector order", type(exc).__name__, exc)
        return [Hit(chunk=c, vector_score=c.score) for c in candidates[:top_k]]
    hits = [
        Hit(chunk=candidates[idx], vector_score=candidates[idx].score, rerank_score=score)
        for idx, score in ranking
    ]
    logger.info(
        "retrieve: %d candidate(s) → %d reranked hit(s) for %r",
        len(candidates), len(hits), question[:60],
    )
    return hits


def retrieve(
    question: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    namespace: str = "",
    rerank: bool = True,
    store: VectorStore | None = None,
    sources: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant results for a question. Sync wrapper — use
    aretrieve() inside an event loop."""
    return run_sync(
        aretrieve(
            question, top_k=top_k, filters=filters, namespace=namespace,
            rerank=rerank, store=store, sources=sources,
        ),
        "aretrieve",
    )
