"""The Source contract — a queryable backend behind retrieve(sources=[...]).

A Source turns a natural-language question into normalized results, whether it
answers from the document corpus or a SQL database. retrieve() fans out over
the selected sources and merges their SourceResults into one envelope, so the
caller never has to know — or branch on — which backend answered.
"""
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The discriminator a caller reads instead of knowing which source answered.
SourceType = Literal["structured", "documents"]


class SourceResult(BaseModel):
    """One normalized result — a database row set, or a document chunk.

    content     — rendered rows or chunk text, ready for an LLM prompt
    source      — the source's name (its key in sources.yaml)
    source_type — "structured" (a database) | "documents" (the corpus)
    provenance  — how to trace it: {sql, params, verified} for SQL,
                  {doc_id, pages, region_ids} for documents
    score       — relevance for ranked document hits; None for exact SQL rows
    raw         — the underlying rows or chunk objects, if the caller wants them
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    content: str
    source: str
    source_type: SourceType
    provenance: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    raw: Any = None


class Source(ABC):
    """A queryable backend behind retrieve(sources=[...]).

    Concrete sources — DocumentSource (the existing corpus) and the SQL sources
    (postgres, mysql, …) — carry their own config and expose one uniform verb:
    answer a question with normalized results. This mirrors the VectorStore
    ABC's "one contract, many backends" shape on the retrieval side.
    """

    #: the source's name — its key in sources.yaml
    name: str

    @abstractmethod
    async def answer(self, question: str, *, top_k: int = 5) -> list[SourceResult]:
        """Answer a natural-language question with normalized results.

        top_k bounds ranked (document) sources; structured sources cap by their
        own configured row_limit and ignore it.
        """

    @abstractmethod
    async def health(self) -> tuple[str, str]:
        """Liveness probe → (status, detail); status is one of ok | warn | fail."""
