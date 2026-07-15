"""MongodbStore — the VectorStore contract on a MongoDB collection.

Hybrid by default: the dense side is Atlas Vector Search ($vectorSearch,
cosine) and the lexical side is Atlas Search ($search) — TRUE BM25 with
lucene english stemming, breadcrumb boosted over body. Queries that carry
the original question text run both and fuse with client-side Reciprocal
Rank Fusion (the server's $rankFusion stage is still Preview); the caller's
reranker produces the final order on top. Lexical failures degrade to
dense-only with a warning.

No query sanitizer needed — the $search `text` operator takes plain text,
not a query language, so "+360%?" is a search, never a syntax error.

Backend quirks handled here so callers never see them:
  - _id is the deterministic "{namespace}:{document_id}:{chunk_id}", and
    re-ingestion deletes the document's rows first — so upserts are
    idempotent AND a re-parse with fewer chunks leaves no orphans
  - search indexes lag writes by a few seconds (eventual consistency) —
    ingestion is fine, but write-then-query-immediately callers must wait
  - dense-only scores are $vectorSearch scores (0–1 for cosine); fused
    scores are RRF ranks — same convention as the other hybrid connectors
"""
import time
from typing import Any

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.mongodb.client import (
    TEXT_INDEX,
    VECTOR_INDEX,
    ensure_collection,
    get_collection,
)
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Fields queries may filter on — declared as filter/token paths in both
# search indexes, so dense and lexical branches accept the same filters.
_FILTERABLE = ("document_id", "category", "section", "kind")

_RRF_K = 60

# $vectorSearch scans this many candidates before applying limit — the
# standard 10-20x overfetch keeps recall high under filters.
_CANDIDATE_FLOOR = 100


def _chunk_key(namespace: str, document_id: str, chunk_id: int) -> str:
    return f"{namespace}:{document_id}:{chunk_id}"


def _breadcrumb(chunk: Chunk, category: str) -> str:
    return " ".join(part for part in (category, chunk.section, chunk.heading) if part)


def _to_document(
    document_id: str,
    chunk: Chunk,
    embedding: list[float],
    category: str,
    namespace: str,
) -> dict[str, Any]:
    """Chunk → MongoDB document (payload keys stringified — BSON requires it)."""
    return {
        "_id": _chunk_key(namespace, document_id, chunk.chunk_id),
        "document_id": document_id,
        "chunk_id": chunk.chunk_id,
        "namespace": namespace,
        "category": category,
        "section": chunk.section,
        "kind": chunk.kind,
        "breadcrumb": _breadcrumb(chunk, category),
        "body": chunk.text or chunk.markdown,
        "embedding": embedding,
        "payload": {
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
        },
    }


def _from_payload(score: float, pl: dict[str, Any]) -> RetrievedChunk:
    """Stored payload → RetrievedChunk (region_ids keys back to int)."""
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


def _check_filters(filters: dict[str, Any] | None) -> None:
    unknown = set(filters or ()) - set(_FILTERABLE)
    if unknown:
        raise ValueError(
            f"unsupported filter field(s) {sorted(unknown)} — the mongodb "
            f"connector filters on {list(_FILTERABLE)}"
        )


def _rrf(dense: list[str], sparse: list[str]) -> list[tuple[str, float]]:
    """Fuse two rank lists — best fused score first, dense wins ties."""
    scores: dict[str, float] = {}
    for ids in (dense, sparse):
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class MongodbStore(VectorStore):
    """Vector storage on a MongoDB collection (search indexes auto-created).

    hybrid=True (default) runs BM25 $search next to every $vectorSearch and
    fuses both signals; hybrid=False is dense-only.
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
        """Replace the document's chunks (delete old rows, insert new ones).

        Returns the chunk count. Deterministic _ids keep the operation
        idempotent, and the delete pass drops orphaned chunk_ids when a
        re-parse yields fewer chunks.
        """
        self._validate_upsert(chunks, embeddings)
        collection = ensure_collection(dimension=len(embeddings[0]))
        t0 = time.perf_counter()
        collection.delete_many({"namespace": namespace, "document_id": document_id})
        collection.insert_many([
            _to_document(document_id, chunk, embedding, category, namespace)
            for chunk, embedding in zip(chunks, embeddings)
        ])
        logger.info(
            "upserted %d chunk(s) for doc %s in %.1fs (search indexes sync within seconds)",
            len(chunks), document_id[:12], time.perf_counter() - t0,
        )
        return len(chunks)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
        text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks, best first; filters are equality constraints.

        When hybrid and `text` is given, a BM25 $search branch runs next to
        $vectorSearch and both fuse with RRF — scores are then RRF ranks.
        """
        _check_filters(filters)
        collection = ensure_collection(dimension=len(vector))
        t0 = time.perf_counter()

        dense = list(collection.aggregate([
            {"$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": max(top_k * 10, _CANDIDATE_FLOOR),
                "limit": top_k,
                "filter": {"namespace": namespace, **(filters or {})},
            }},
            {"$project": {"payload": 1, "score": {"$meta": "vectorSearchScore"}}},
        ]))

        sparse: list[dict[str, Any]] = []
        if self.hybrid and text and text.strip():
            try:
                sparse = list(collection.aggregate([
                    {"$search": {
                        "index": TEXT_INDEX,
                        "compound": {
                            # breadcrumb outranks body on equal matches —
                            # same weighting the other connectors use
                            "should": [
                                {"text": {"query": text, "path": "breadcrumb",
                                          "score": {"boost": {"value": 2.0}}}},
                                {"text": {"query": text, "path": "body"}},
                            ],
                            "minimumShouldMatch": 1,
                            "filter": [
                                {"equals": {"path": "namespace", "value": namespace}},
                                *(
                                    {"equals": {"path": key, "value": value}}
                                    for key, value in (filters or {}).items()
                                ),
                            ],
                        },
                    }},
                    {"$limit": top_k},
                    {"$project": {"payload": 1}},
                ]))
            except Exception as exc:
                logger.warning(
                    "BM25 branch failed (%s: %s) — dense-only query",
                    type(exc).__name__, exc,
                )

        payloads = {doc["_id"]: doc["payload"] for doc in dense}
        payloads.update({doc["_id"]: doc["payload"] for doc in sparse})
        if sparse:
            ranked = _rrf(
                [doc["_id"] for doc in dense], [doc["_id"] for doc in sparse]
            )[:top_k]
        else:
            ranked = [(doc["_id"], float(doc["score"])) for doc in dense]

        hits = [_from_payload(score, payloads[doc_id]) for doc_id, score in ranked]
        logger.info(
            "query returned %d hit(s) in %.2fs (top_k=%d, hybrid=%s, filters=%s)",
            len(hits), time.perf_counter() - t0, top_k, bool(sparse),
            sorted(filters) if filters else None,
        )
        return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's chunks. Returns the exact count removed."""
        result = get_collection().delete_many(
            {"namespace": namespace, "document_id": document_id}
        )
        logger.info("deleted %d chunk(s) for doc %s", result.deleted_count, document_id[:12])
        return result.deleted_count
