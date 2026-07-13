"""Vector-store contract — the interface every connector implements.

ingestlib works with any vector database: a connector (pinecone, qdrant,
milvus, pgvector, ...) subclasses VectorStore and handles its backend's
quirks — ID schemes, metadata encoding, deletion semantics — behind these
three methods, so pipelines are written once and run against whichever
database is configured.

Principle: vectors in, records out. Dense embedding happens outside the store
(via foundations.llm.embed_text on chunk.embedding_text), so connectors stay
provider-agnostic; a connector with a lexical side may compute its own sparse
form internally from the chunk text.
"""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ingestlib.operations.split.models import Chunk


class RetrievedChunk(BaseModel):
    """One query hit — a stored chunk restored with its retrieval score.

    Carries everything needed to answer AND cite: content (markdown/text),
    location (document_id, pages, region_ids → bboxes via the artifact store),
    and context (section, heading, category).
    """

    model_config = ConfigDict(frozen=True)

    score: float
    document_id: str
    chunk_id: int
    section: str = ""
    heading: str = ""
    markdown: str = ""
    text: str = ""
    pages: list[int] = Field(default_factory=list)
    region_ids: dict[int, list[int]] = Field(default_factory=dict)
    category: str = ""
    kind: str = "text"


class VectorStore(ABC):
    """Contract for pushing split chunks into a vector database and querying them.

    Implementations must make upserts idempotent per (document_id, chunk_id) —
    re-ingesting a document overwrites its vectors, never duplicates them.
    """

    @abstractmethod
    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        category: str = "",
        namespace: str = "",
    ) -> int:
        """Store one embedding per chunk with full provenance payload.

        Returns the number of vectors written. embeddings[i] belongs to
        chunks[i]; use _validate_upsert() to enforce the pairing. `category`
        is the document-type label (from classify) stored on every vector so
        queries can filter by it.
        """

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
        text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks to `vector`, best first.

        filters are equality constraints on payload fields, e.g.
        {"category": "research_paper", "section": "methods"}.
        `text` is the original query text — connectors with a lexical/hybrid
        side use it for sparse search; dense-only connectors ignore it.
        """

    @abstractmethod
    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Remove every vector belonging to a document. Returns count removed."""

    @staticmethod
    def _validate_upsert(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Shared guard: chunks and embeddings must pair up 1:1 and be non-empty."""
        if not chunks:
            raise ValueError("chunks must contain at least one item")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must pair 1:1 — got {len(chunks)} chunks "
                f"and {len(embeddings)} embeddings"
            )
        dims = {len(e) for e in embeddings}
        if len(dims) > 1:
            raise ValueError(f"embeddings have inconsistent dimensions: {sorted(dims)}")
