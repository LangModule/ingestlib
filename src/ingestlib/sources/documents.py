"""DocumentSource — the existing document corpus, as a Source.

Wraps vector search + rerank so documents compose in the same
retrieve(sources=[...]) fan-out as databases. answer() runs the exact
dense+rerank path retrieve() already uses (no new logic), mapping each Hit into
a SourceResult so the caller reads one shape whatever answered.
"""
from typing import Any

from ingestlib.sources.base import Source, SourceResult
from ingestlib.storage import VectorStore


class DocumentSource(Source):
    """The ingested document corpus (or one namespace of it) as a retrieval Source."""

    def __init__(
        self,
        name: str,
        *,
        namespace: str = "",
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        store: VectorStore | None = None,
    ) -> None:
        self.name = name
        self._namespace = namespace
        self._filters = filters
        self._rerank = rerank
        self._store = store

    async def answer(self, question: str, *, top_k: int = 5) -> list[SourceResult]:
        # imported here, not at module top: retriever imports the registry lazily,
        # so calling back into it this way avoids an import cycle.
        from ingestlib.services.retrieve.retriever import _retrieve_document_hits

        hits = await _retrieve_document_hits(
            question, top_k=top_k, filters=self._filters,
            namespace=self._namespace, rerank=self._rerank, store=self._store,
        )
        return [SourceResult(
            content=f"{h.citation}\n{h.chunk.markdown or h.chunk.text}",
            source=self.name,
            source_type="documents",
            provenance={
                "document_id": h.chunk.document_id,
                "pages": h.chunk.pages,
                "region_ids": h.chunk.region_ids,
            },
            score=h.rerank_score if h.rerank_score is not None else h.vector_score,
            raw=h,
        ) for h in hits]

    async def health(self) -> tuple[str, str]:
        from ingestlib.storage import default_store

        try:
            self._store or default_store()
        except Exception as exc:
            return "fail", f"source {self.name}: {exc}"
        return "ok", f"source {self.name} (documents): ready"
