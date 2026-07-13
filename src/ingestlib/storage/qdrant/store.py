"""QdrantStore — the VectorStore contract implemented on a Qdrant collection.

Backend quirks handled here so callers never see them:
  - point IDs must be UUIDs or unsigned ints, not arbitrary strings →
    deterministic uuid5 of "{document_id}:{chunk_id}", so re-ingestion still
    overwrites in place
  - payloads are native JSON (no flattening needed), but JSON object keys are
    strings → region_ids page numbers round-trip through str and back to int
  - Qdrant has no namespaces → the namespace is a payload field every query
    and deletion filters on, which mirrors the isolation semantics
  - deletion works by filter directly (no ID listing dance); the count comes
    from an exact count call before deleting

Dense-only: the `text` query param is accepted and ignored (a sparse/hybrid
side is a connector v2, same shape as the Pinecone one).
"""
import time
import uuid
from typing import Any

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct

from ingestlib.config import get_qdrant_config
from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.qdrant.client import ensure_collection, get_qdrant_client
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_UPSERT_BATCH = 100

# Fixed namespace for uuid5 so "{document_id}:{chunk_id}" always maps to the
# same point ID across processes and runs.
_POINT_ID_NAMESPACE = uuid.UUID("6e6763a2-9a1b-4a3e-9c1f-8d2e5b7c4f01")


def _point_id(document_id: str, chunk_id: int) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{document_id}:{chunk_id}"))


def _to_payload(
    document_id: str, chunk: Chunk, category: str, namespace: str
) -> dict[str, Any]:
    """Chunk → Qdrant payload (native JSON; region_ids keys stringified)."""
    return {
        "document_id": document_id,
        "chunk_id": chunk.chunk_id,
        "section": chunk.section,
        "heading": chunk.heading,
        "kind": chunk.kind,
        "category": category,
        "token_estimate": chunk.token_estimate,
        "pages": chunk.pages,
        "region_ids": {str(k): v for k, v in chunk.region_ids.items()},
        "markdown": chunk.markdown,
        "text": chunk.text,
        "namespace": namespace,
    }


def _from_point(point: Any) -> RetrievedChunk:
    """Qdrant scored point → RetrievedChunk (region_ids keys back to int)."""
    pl = point.payload or {}
    return RetrievedChunk(
        score=float(point.score),
        document_id=pl["document_id"],
        chunk_id=int(pl["chunk_id"]),
        section=pl.get("section", ""),
        heading=pl.get("heading", ""),
        markdown=pl.get("markdown", ""),
        text=pl.get("text", ""),
        pages=[int(p) for p in pl.get("pages", [])],
        region_ids={int(k): [int(i) for i in v] for k, v in pl.get("region_ids", {}).items()},
        category=pl.get("category", ""),
        kind=pl.get("kind", "text"),
    )


def _filter(
    namespace: str, filters: dict[str, Any] | None = None, document_id: str | None = None
) -> Filter:
    """Equality conditions on payload fields — namespace always included."""
    must = [FieldCondition(key="namespace", match=MatchValue(value=namespace))]
    for key, value in (filters or {}).items():
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    if document_id is not None:
        must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
    return Filter(must=must)


class QdrantStore(VectorStore):
    """Vector storage on a Qdrant collection (auto-created on first use)."""

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        category: str = "",
        namespace: str = "",
    ) -> int:
        """Store one point per chunk; deterministic IDs overwrite in place."""
        self._validate_upsert(chunks, embeddings)
        collection = ensure_collection(dimension=len(embeddings[0]))
        client = get_qdrant_client()

        points = [
            PointStruct(
                id=_point_id(document_id, chunk.chunk_id),
                vector=embedding,
                payload=_to_payload(document_id, chunk, category, namespace),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        t0 = time.perf_counter()
        for i in range(0, len(points), _UPSERT_BATCH):
            client.upsert(collection_name=collection, points=points[i : i + _UPSERT_BATCH])
        logger.info(
            "upserted %d point(s) for doc %s in %.1fs",
            len(points), document_id[:12], time.perf_counter() - t0,
        )
        return len(points)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
        text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks, best first; filters are payload equality constraints.

        `text` is ignored — this connector is dense-only.
        """
        collection = ensure_collection(dimension=len(vector))
        t0 = time.perf_counter()
        response = get_qdrant_client().query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            query_filter=_filter(namespace, filters),
            with_payload=True,
        )
        hits = [_from_point(p) for p in response.points]
        logger.info(
            "query returned %d hit(s) in %.2fs (top_k=%d, filters=%s)",
            len(hits), time.perf_counter() - t0, top_k, sorted(filters) if filters else None,
        )
        return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's points by filter. Returns count removed."""
        client = get_qdrant_client()
        collection = get_qdrant_config().collection_name
        if not client.collection_exists(collection):
            return 0  # nothing was ever stored
        doc_filter = _filter(namespace, document_id=document_id)
        count = client.count(
            collection_name=collection, count_filter=doc_filter, exact=True
        ).count
        if count:
            client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=doc_filter),
            )
        logger.info("deleted %d point(s) for doc %s", count, document_id[:12])
        return count
