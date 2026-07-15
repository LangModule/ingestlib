"""MilvusStore — the VectorStore contract on a Milvus collection.

Hybrid by default: the dense side is cosine ANN over the embedding field and
the lexical side is the server's built-in BM25 over `search_text` (breadcrumb
prepended to body, so heading-path tokens weigh in). Queries that carry the
original question text run both in ONE hybrid_search call — the server fuses
them with Reciprocal Rank Fusion — and the caller's reranker produces the
final order on top. Sparse failures degrade to a dense-only search with a
warning.

No query sanitizer needed — BM25 search takes raw text through the server's
analyzer, so "+360%?" is a search, never a syntax error.

Backend quirks handled here so callers never see them:
  - ids are the deterministic "{namespace}:{document_id}:{chunk_id}", and
    re-ingestion deletes the document's rows first — idempotent, and a
    re-parse with fewer chunks leaves no orphans
  - filters are boolean expression strings, so values are escaped before
    interpolation and filter KEYS are validated against the schema
  - dense-only scores are cosine similarity; fused scores are RRF ranks —
    same convention as the other hybrid connectors
"""
import time
from typing import Any

from pymilvus import AnnSearchRequest, RRFRanker

from ingestlib.config import get_milvus_config
from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.milvus.client import ensure_collection, get_milvus_client
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Fields queries may filter on — scalar columns in the schema, valid in both
# the dense and sparse branches' filter expressions.
_FILTERABLE = ("document_id", "category", "section", "kind")

_RRF_K = 60

_UPSERT_BATCH = 500


def _chunk_key(namespace: str, document_id: str, chunk_id: int) -> str:
    return f"{namespace}:{document_id}:{chunk_id}"


def _breadcrumb(chunk: Chunk, category: str) -> str:
    return " ".join(part for part in (category, chunk.section, chunk.heading) if part)


def _escape(value: str) -> str:
    """Escape a string for interpolation into a Milvus filter expression."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _expr(
    namespace: str, filters: dict[str, Any] | None = None, document_id: str | None = None
) -> str:
    """Equality conditions as a filter expression — namespace always included."""
    unknown = set(filters or ()) - set(_FILTERABLE)
    if unknown:
        raise ValueError(
            f"unsupported filter field(s) {sorted(unknown)} — the milvus "
            f"connector filters on {list(_FILTERABLE)}"
        )
    clauses = [f'namespace == "{_escape(namespace)}"']
    for key, value in (filters or {}).items():
        clauses.append(f'{key} == "{_escape(str(value))}"')
    if document_id is not None:
        clauses.append(f'document_id == "{_escape(document_id)}"')
    return " and ".join(clauses)


def _to_row(
    document_id: str,
    chunk: Chunk,
    embedding: list[float],
    category: str,
    namespace: str,
) -> dict[str, Any]:
    """Chunk → Milvus row (sparse is computed server-side from search_text)."""
    return {
        "id": _chunk_key(namespace, document_id, chunk.chunk_id),
        "document_id": document_id,
        "chunk_id": chunk.chunk_id,
        "namespace": namespace,
        "category": category,
        "section": chunk.section,
        "kind": chunk.kind,
        "search_text": f"{_breadcrumb(chunk, category)}\n{chunk.text or chunk.markdown}",
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


def _from_hit(hit: dict[str, Any]) -> RetrievedChunk:
    """Milvus search hit → RetrievedChunk (region_ids keys back to int)."""
    pl = hit["entity"]["payload"]
    return RetrievedChunk(
        score=float(hit["distance"]),
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


class MilvusStore(VectorStore):
    """Vector storage on a Milvus collection (auto-created on first use).

    hybrid=True (default) runs the server's BM25 next to every dense search,
    fused server-side with RRF; hybrid=False is dense-only.
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
        """Replace the document's rows (delete old, insert new — raw text in,
        sparse vectors computed server-side).

        Returns the chunk count; deterministic ids plus the delete pass keep
        re-ingestion idempotent and orphan-free.
        """
        self._validate_upsert(chunks, embeddings)
        collection = ensure_collection(dimension=len(embeddings[0]))
        client = get_milvus_client()
        t0 = time.perf_counter()
        client.delete(collection, filter=_expr(namespace, document_id=document_id))
        rows = [
            _to_row(document_id, chunk, embedding, category, namespace)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        for i in range(0, len(rows), _UPSERT_BATCH):
            client.insert(collection, rows[i : i + _UPSERT_BATCH])
        logger.info(
            "upserted %d chunk(s) for doc %s in %.1fs",
            len(rows), document_id[:12], time.perf_counter() - t0,
        )
        return len(rows)

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
        server call fused with RRF — scores are then RRF ranks, not cosine.
        """
        expr = _expr(namespace, filters)
        collection = ensure_collection(dimension=len(vector))
        client = get_milvus_client()
        t0 = time.perf_counter()

        fused = False
        if self.hybrid and text and text.strip():
            try:
                results = client.hybrid_search(
                    collection,
                    reqs=[
                        AnnSearchRequest(data=[vector], anns_field="embedding",
                                         param={}, limit=top_k, expr=expr),
                        AnnSearchRequest(data=[text], anns_field="sparse",
                                         param={}, limit=top_k, expr=expr),
                    ],
                    ranker=RRFRanker(_RRF_K),
                    limit=top_k,
                    output_fields=["payload"],
                )
                fused = True
            except Exception as exc:
                logger.warning(
                    "hybrid search failed (%s: %s) — dense-only query",
                    type(exc).__name__, exc,
                )
                results = None
        else:
            results = None

        if results is None:
            results = client.search(
                collection, data=[vector], anns_field="embedding",
                limit=top_k, filter=expr, output_fields=["payload"],
            )

        hits = [_from_hit(h) for h in results[0]]
        logger.info(
            "query returned %d hit(s) in %.2fs (top_k=%d, hybrid=%s, filters=%s)",
            len(hits), time.perf_counter() - t0, top_k, fused,
            sorted(filters) if filters else None,
        )
        return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's rows by filter. Returns count removed."""
        client = get_milvus_client()
        collection = get_milvus_config().collection_name
        if not client.has_collection(collection):
            return 0  # nothing was ever stored
        result = client.delete(collection, filter=_expr(namespace, document_id=document_id))
        count = int(result["delete_count"])
        logger.info("deleted %d chunk(s) for doc %s", count, document_id[:12])
        return count
