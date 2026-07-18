"""WeaviateStore — the VectorStore contract on a Weaviate collection.

Hybrid by default: the dense side is HNSW cosine over the caller-supplied
vector and the lexical side is the server's native BM25 over breadcrumb +
body, the breadcrumb boosted over the body. Queries that carry the original
question text run both signals in ONE call — the server fuses them with
ranked (RRF-style) fusion — and the caller's reranker produces the final
order on top. Hybrid failures degrade to a dense-only search with a warning.

No query sanitizer needed — BM25 takes plain text through the server's
analyzer, so "+360%?" is a search, never a syntax error.

Backend quirks handled here so callers never see them:
  - object IDs must be UUIDs, not arbitrary strings → deterministic uuid5
    of "{namespace}:{document_id}:{chunk_id}", and re-ingestion deletes the
    document's objects first — idempotent, and a re-parse with fewer chunks
    leaves no orphans
  - properties are typed, so the full chunk payload rides as one JSON
    string (region_ids page keys round-trip through str and back to int)
  - Weaviate has no namespaces → the namespace is a field-tokenized
    property every query and deletion filters on
  - dense-only scores are cosine similarity (1 - distance); fused scores
    are the server's hybrid scores — rank blends, same convention as the
    other hybrid connectors
"""
import json
import time
import uuid
from typing import Any

from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.weaviate.client import (
    collection_name,
    ensure_collection,
    get_weaviate_client,
)
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Fields queries may filter on — field-tokenized properties, valid in both
# the dense and hybrid branches' filters.
_FILTERABLE = ("document_id", "category", "section", "kind")

_UPSERT_BATCH = 100

# Fixed namespace for uuid5 so the same chunk key always maps to the same
# object ID across processes and runs.
_OBJECT_ID_NAMESPACE = uuid.UUID("f3b1a6c8-2d4e-4f7a-9b0c-1e5d8a7f6b3c")


def _object_id(document_id: str, chunk_id: int, namespace: str = "") -> str:
    return str(uuid.uuid5(_OBJECT_ID_NAMESPACE, f"{namespace}:{document_id}:{chunk_id}"))


def _breadcrumb(chunk: Chunk, category: str) -> str:
    return " ".join(part for part in (category, chunk.section, chunk.heading) if part)


def _to_properties(
    document_id: str, chunk: Chunk, category: str, namespace: str
) -> dict[str, Any]:
    """Chunk → Weaviate properties (payload as JSON; region_ids keys stringified)."""
    return {
        "document_id": document_id,
        "namespace": namespace,
        "category": category,
        "section": chunk.section,
        "kind": chunk.kind,
        "breadcrumb": _breadcrumb(chunk, category),
        "body": chunk.text or chunk.markdown,
        "payload": json.dumps({
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
        }),
    }


def _from_payload(score: float, payload_json: str) -> RetrievedChunk:
    """Stored payload JSON → RetrievedChunk (region_ids keys back to int)."""
    pl = json.loads(payload_json)
    return RetrievedChunk(
        score=score,
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
    """Equality conditions on properties — namespace always included."""
    unknown = set(filters or ()) - set(_FILTERABLE)
    if unknown:
        raise ValueError(
            f"unsupported filter field(s) {sorted(unknown)} — the weaviate "
            f"connector filters on {list(_FILTERABLE)}"
        )
    conditions = [Filter.by_property("namespace").equal(namespace)]
    for key, value in (filters or {}).items():
        conditions.append(Filter.by_property(key).equal(value))
    if document_id is not None:
        conditions.append(Filter.by_property("document_id").equal(document_id))
    return Filter.all_of(conditions)


class WeaviateStore(VectorStore):
    """Vector storage on a Weaviate collection (auto-created on first use).

    hybrid=True (default) runs the server's BM25 next to every dense search,
    fused server-side; hybrid=False is dense-only.
    """

    def __init__(self, hybrid: bool = True):
        self.hybrid = hybrid

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        category: str = "",
        namespace: str = "",
    ) -> int:
        """Replace the document's objects (delete old, insert new).

        Returns the chunk count; deterministic object IDs plus the delete
        pass keep re-ingestion idempotent and orphan-free.
        """
        self._validate_upsert(chunks, embeddings)
        collection = get_weaviate_client().collections.get(
            ensure_collection(dimension=len(embeddings[0]))
        )
        t0 = time.perf_counter()
        collection.data.delete_many(
            where=_filter(namespace, document_id=document_id)
        )
        objects = [
            DataObject(
                uuid=_object_id(document_id, chunk.chunk_id, namespace),
                properties=_to_properties(document_id, chunk, category, namespace),
                vector=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        for i in range(0, len(objects), _UPSERT_BATCH):
            collection.data.insert_many(objects[i : i + _UPSERT_BATCH])
        logger.info(
            "upserted %d object(s) for doc %s in %.1fs",
            len(objects), document_id[:12], time.perf_counter() - t0,
        )
        return len(objects)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
        text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks, best first; filters are equality constraints.

        When hybrid and `text` is given, dense and BM25 branches run in one
        server call with ranked fusion — scores are then fused ranks, not
        cosine.
        """
        query_filter = _filter(namespace, filters)
        collection = get_weaviate_client().collections.get(
            ensure_collection(dimension=len(vector))
        )
        t0 = time.perf_counter()

        fused = False
        response = None
        if self.hybrid and text and text.strip():
            try:
                response = collection.query.hybrid(
                    query=text,
                    vector=vector,
                    # breadcrumb outranks body on equal matches — same
                    # weighting the other connectors use
                    query_properties=["breadcrumb^2", "body"],
                    fusion_type=HybridFusion.RANKED,
                    limit=top_k,
                    filters=query_filter,
                    return_metadata=MetadataQuery(score=True),
                )
                fused = True
            except Exception as exc:
                logger.warning(
                    "hybrid search failed (%s: %s) — dense-only query",
                    type(exc).__name__, exc,
                )

        if response is None:
            response = collection.query.near_vector(
                near_vector=vector,
                limit=top_k,
                filters=query_filter,
                return_metadata=MetadataQuery(distance=True),
            )

        hits = []
        for obj in response.objects:
            if fused:
                score = float(obj.metadata.score or 0.0)
            else:
                score = 1.0 - float(obj.metadata.distance or 0.0)
            hits.append(_from_payload(score, obj.properties["payload"]))
        logger.info(
            "query returned %d hit(s) in %.2fs (top_k=%d, hybrid=%s, filters=%s)",
            len(hits), time.perf_counter() - t0, top_k, fused,
            sorted(filters) if filters else None,
        )
        return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's objects by filter. Returns count removed."""
        client = get_weaviate_client()
        name = collection_name()
        if not client.collections.exists(name):
            return 0  # nothing was ever stored
        result = client.collections.get(name).data.delete_many(
            where=_filter(namespace, document_id=document_id)
        )
        count = int(result.successful)
        logger.info("deleted %d object(s) for doc %s", count, document_id[:12])
        return count
