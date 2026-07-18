"""OpensearchStore — the VectorStore contract on an OpenSearch index.

Hybrid by default: the dense side is faiss HNSW k-NN over the embedding
field and the lexical side is Lucene BM25 over breadcrumb + body, the
breadcrumb boosted over the body. Queries that carry the original question
text run both and fuse with client-side Reciprocal Rank Fusion; the
caller's reranker produces the final order on top. Lexical failures degrade
to dense-only with a warning.

No query sanitizer needed — the match query analyzes plain text, so
"+360%?" is a search, never a syntax error.

Backend quirks handled here so callers never see them:
  - _id is the deterministic "{namespace}:{document_id}:{chunk_id}", and
    re-ingestion deletes the document's rows first — so upserts are
    idempotent AND a re-parse with fewer chunks leaves no orphans
  - writes become searchable on index refresh, so upserts and deletes
    refresh explicitly — write-then-query works without waiting
  - the k-NN filter rides INSIDE the knn clause (faiss efficient
    filtering); a post-filter would let filtered-out hits consume k
  - dense-only scores are the engine's k-NN similarity (higher is
    better); fused scores are RRF ranks — same convention as the other
    hybrid connectors
"""
import time
from typing import Any

from opensearchpy.helpers import bulk

from ingestlib.config import get_opensearch_config
from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.opensearch.client import ensure_index, get_opensearch_client
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Fields queries may filter on — keyword fields in the mapping, valid in
# both the dense and lexical branches' filter clauses.
_FILTERABLE = ("document_id", "category", "section", "kind")

_RRF_K = 60


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
    """Chunk → OpenSearch document (payload keys stringified — JSON requires it)."""
    return {
        "document_id": document_id,
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


def _filter_terms(
    namespace: str, filters: dict[str, Any] | None = None, document_id: str | None = None
) -> list[dict[str, Any]]:
    """Equality conditions as term clauses — namespace always included."""
    unknown = set(filters or ()) - set(_FILTERABLE)
    if unknown:
        raise ValueError(
            f"unsupported filter field(s) {sorted(unknown)} — the opensearch "
            f"connector filters on {list(_FILTERABLE)}"
        )
    terms = [{"term": {"namespace": namespace}}]
    for key, value in (filters or {}).items():
        terms.append({"term": {key: value}})
    if document_id is not None:
        terms.append({"term": {"document_id": document_id}})
    return terms


def _rrf(dense: list[str], sparse: list[str]) -> list[tuple[str, float]]:
    """Fuse two rank lists — best fused score first, dense wins ties."""
    scores: dict[str, float] = {}
    for ids in (dense, sparse):
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class OpensearchStore(VectorStore):
    """Vector storage on an OpenSearch index (auto-created on first use).

    hybrid=True (default) runs BM25 next to every k-NN search and fuses
    both signals; hybrid=False is dense-only.
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
        """Replace the document's rows (delete old, bulk-insert new).

        Returns the chunk count. Deterministic _ids keep the operation
        idempotent, and the delete pass drops orphaned chunk_ids when a
        re-parse yields fewer chunks.
        """
        self._validate_upsert(chunks, embeddings)
        index = ensure_index(dimension=len(embeddings[0]))
        client = get_opensearch_client()
        t0 = time.perf_counter()
        doc_terms = _filter_terms(namespace, document_id=document_id)
        client.delete_by_query(
            index=index,
            body={"query": {"bool": {"filter": doc_terms}}},
            refresh=True,
        )
        bulk(client, [
            {
                "_op_type": "index",
                "_index": index,
                "_id": _chunk_key(namespace, document_id, chunk.chunk_id),
                "_source": _to_document(document_id, chunk, embedding, category, namespace),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ])
        client.indices.refresh(index=index)  # searchable immediately, no interval wait
        logger.info(
            "upserted %d chunk(s) for doc %s in %.1fs",
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

        When hybrid and `text` is given, a BM25 branch runs next to the k-NN
        and both fuse with RRF — scores are then RRF ranks, not similarities.
        """
        terms = _filter_terms(namespace, filters)
        index = ensure_index(dimension=len(vector))
        client = get_opensearch_client()
        t0 = time.perf_counter()

        dense = client.search(index=index, body={
            "size": top_k,
            "_source": ["payload"],
            "query": {"knn": {"embedding": {
                "vector": vector, "k": top_k,
                "filter": {"bool": {"filter": terms}},
            }}},
        })["hits"]["hits"]

        sparse: list[dict[str, Any]] = []
        if self.hybrid and text and text.strip():
            try:
                sparse = client.search(index=index, body={
                    "size": top_k,
                    "_source": ["payload"],
                    "query": {"bool": {
                        # breadcrumb outranks body on equal matches — same
                        # weighting the other connectors use
                        "must": [{"multi_match": {
                            "query": text, "fields": ["breadcrumb^2", "body"],
                        }}],
                        "filter": terms,
                    }},
                })["hits"]["hits"]
            except Exception as exc:
                logger.warning(
                    "BM25 branch failed (%s: %s) — dense-only query",
                    type(exc).__name__, exc,
                )

        payloads = {hit["_id"]: hit["_source"]["payload"] for hit in dense}
        payloads.update({hit["_id"]: hit["_source"]["payload"] for hit in sparse})
        if sparse:
            ranked = _rrf(
                [hit["_id"] for hit in dense], [hit["_id"] for hit in sparse]
            )[:top_k]
        else:
            ranked = [(hit["_id"], float(hit["_score"])) for hit in dense]

        hits = [_from_payload(score, payloads[doc_id]) for doc_id, score in ranked]
        logger.info(
            "query returned %d hit(s) in %.2fs (top_k=%d, hybrid=%s, filters=%s)",
            len(hits), time.perf_counter() - t0, top_k, bool(sparse),
            sorted(filters) if filters else None,
        )
        return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's rows. Returns the exact count removed."""
        client = get_opensearch_client()
        index = get_opensearch_config().index_name
        if not client.indices.exists(index=index):
            return 0  # nothing was ever stored
        doc_terms = _filter_terms(namespace, document_id=document_id)
        response = client.delete_by_query(
            index=index,
            body={"query": {"bool": {"filter": doc_terms}}},
            refresh=True,
        )
        count = int(response["deleted"])
        logger.info("deleted %d chunk(s) for doc %s", count, document_id[:12])
        return count
