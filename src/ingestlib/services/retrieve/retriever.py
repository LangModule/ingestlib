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
    collection: str | None = None,
    min_confidence: float | None = None,
    kinds: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant results for a question (async).

    question       — natural-language query
    top_k          — results to return per source
    filters        — vector-payload constraints, e.g. {"category": "x", "kind": "table"}
    rerank         — rerank document candidates with config.yaml's `reranker`
    store          — vector store connector; defaults to config.yaml's `vector_store`
    collection     — registry-backed filter: keep only hits whose document is in this
                     collection (consults the registry; None = no collection filter)
    min_confidence — registry-backed filter: drop hits whose document's classify
                     confidence is below this (None = no confidence filter)
    kinds          — keep only chunks of these content kinds (text | table | figure |
                     mixed) — e.g. ["table", "figure"] for chunks carrying a table or
                     chart/figure; None = every kind
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
        rerank=rerank, store=store, collection=collection, min_confidence=min_confidence,
        kinds=kinds,
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
    collection: str | None = None,
    min_confidence: float | None = None,
    kinds: list[str] | None = None,
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

    # content-kind filter: chunk.kind rides in the vector payload, so this needs
    # no registry round-trip (text | table | figure | mixed)
    if kinds:
        wanted = set(kinds)
        candidates = [c for c in candidates if c.kind in wanted]
        if not candidates:
            logger.info("retrieve: kind filter %s removed all hits for %r", sorted(wanted), question[:60])
            return []

    # registry-backed filters: keep only candidates whose DOCUMENT qualifies
    # (touched only when a registry filter is requested — basic retrieve stays
    # vector-store-driven).
    if collection is not None or min_confidence is not None:
        from ingestlib.storage import registry

        attrs = await asyncio.to_thread(
            registry.document_attrs, list({c.document_id for c in candidates})
        )

        def _qualifies(c: Any) -> bool:
            a = attrs.get(c.document_id, {})
            if collection is not None and a.get("collection") != collection:
                return False
            if min_confidence is not None and (a.get("classify_confidence") or 0.0) < min_confidence:
                return False
            return True

        candidates = [c for c in candidates if _qualifies(c)]
        if not candidates:
            logger.info("retrieve: registry filters removed all hits for %r", question[:60])
            return []

    if not use_rerank or len(candidates) == 1:
        return await _enrich_hits([Hit(chunk=c, vector_score=c.score) for c in candidates[:top_k]])

    documents = [c.markdown or c.text for c in candidates]
    try:
        ranking = await _reranker(reranker)(question, documents, top_n=top_k)
    except Exception as exc:
        # retrieval must not die because the reranker hiccuped — degrade to
        # vector order and say so loudly
        logger.warning("rerank failed (%s: %s) — returning vector order", type(exc).__name__, exc)
        return await _enrich_hits([Hit(chunk=c, vector_score=c.score) for c in candidates[:top_k]])
    hits = [
        Hit(chunk=candidates[idx], vector_score=candidates[idx].score, rerank_score=score)
        for idx, score in ranking
    ]
    logger.info(
        "retrieve: %d candidate(s) → %d reranked hit(s) for %r",
        len(candidates), len(hits), question[:60],
    )
    return await _enrich_hits(hits)


async def _enrich_hits(hits: list[Hit]) -> list[Hit]:
    """Attach each hit's document-level collection + classify confidence from the
    registry. Best-effort: retrieval degrades to un-enriched hits if the registry
    is unreachable — the chunk still carries category and full provenance."""
    if not hits:
        return hits
    from ingestlib.storage import registry

    try:
        attrs = await asyncio.to_thread(
            registry.document_attrs, list({h.chunk.document_id for h in hits})
        )
    except Exception as exc:
        logger.warning("hit enrichment skipped (registry: %s) — hits un-enriched", exc)
        return hits
    return [
        h.model_copy(update={
            "collection": (attrs.get(h.chunk.document_id) or {}).get("collection") or "",
            "confidence": (attrs.get(h.chunk.document_id) or {}).get("classify_confidence"),
        })
        for h in hits
    ]


def retrieve(
    question: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    namespace: str = "",
    rerank: bool = True,
    store: VectorStore | None = None,
    sources: list[str] | None = None,
    collection: str | None = None,
    min_confidence: float | None = None,
    kinds: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve the most relevant results for a question. Sync wrapper — use
    aretrieve() inside an event loop."""
    return run_sync(
        aretrieve(
            question, top_k=top_k, filters=filters, namespace=namespace,
            rerank=rerank, store=store, sources=sources,
            collection=collection, min_confidence=min_confidence, kinds=kinds,
        ),
        "aretrieve",
    )
