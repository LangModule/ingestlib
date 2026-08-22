"""Data models returned by retrieve(): Hit and RetrievalResult."""
from pydantic import BaseModel, ConfigDict, Field, computed_field

from ingestlib.sources.base import SourceResult
from ingestlib.storage.base import RetrievedChunk


class Hit(BaseModel):
    """One retrieved chunk with both scoring signals.

    vector_score — the store's retrieval score: cosine similarity on dense
                   queries, an RRF rank score on fused hybrid queries
    rerank_score — reranker relevance (None when reranking was off)
    """

    model_config = ConfigDict(frozen=True)

    chunk: RetrievedChunk
    vector_score: float
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        """Human-readable source pointer, e.g. 'doc 7b6b95d79149 · p.4 · methods'."""
        pages = ",".join(str(p) for p in self.chunk.pages) or "?"
        return f"doc {self.chunk.document_id[:12]} · p.{pages} · {self.chunk.section}"


class RetrievalResult(BaseModel):
    """Ranked results for one question, ready for prompt building.

    hits    — document chunks (the default, sources-free retrieve)
    results — normalized items when retrieve(sources=[...]) fanned out over
              documents AND/OR databases; each carries its own source + provenance
    """

    model_config = ConfigDict(frozen=True)

    question: str
    hits: list[Hit] = Field(default_factory=list)
    results: list[SourceResult] = Field(default_factory=list)

    @computed_field
    @property
    def context(self) -> str:
        """Numbered, cited context — paste-ready as LLM context. Renders the
        mixed-source `results` when a sources= fan-out produced them, else the
        document `hits`."""
        if self.results:
            return "\n\n".join(
                f"[{i}] ({r.source}) {r.content}"
                for i, r in enumerate(self.results, start=1)
            )
        return "\n\n".join(
            f"[{i}] ({h.citation})\n{h.chunk.markdown or h.chunk.text}"
            for i, h in enumerate(self.hits, start=1)
        )
