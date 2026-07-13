"""PineconeStore — the VectorStore contract implemented on Pinecone serverless.

Hybrid by default: every chunk lands in the dense index (Nova embedding,
passed in by the caller) AND the sparse index (hosted lexical embedding,
computed here from chunk text). Queries that carry the original question text
search both and merge — dense hits first, sparse-only hits appended — with
final ordering left to the caller's reranker, which sidesteps the fact that
cosine and dotproduct scores are not comparable. Sparse failures degrade to
dense-only with a warning; retrieval never dies because the lexical half
hiccuped.

Backend quirks handled here so callers never see them:
  - vector IDs are "{document_id}:{chunk_id}" → re-ingestion overwrites in place
  - metadata must be flat: pages become a list of strings, region_ids a JSON
    string; both are restored on read
  - serverless cannot delete by metadata filter → deletion lists IDs by the
    document prefix and deletes by ID batch
"""
import json
import time
from typing import Any

from ingestlib.config import get_pinecone_config
from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.pinecone.client import (
    embed_sparse,
    ensure_index,
    ensure_sparse_index,
    get_pinecone_client,
)
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_UPSERT_BATCH = 100


def _vector_id(document_id: str, chunk_id: int) -> str:
    return f"{document_id}:{chunk_id}"


def _to_metadata(document_id: str, chunk: Chunk, category: str) -> dict[str, Any]:
    """Chunk → flat Pinecone metadata (lists of strings, JSON-encoded dicts)."""
    return {
        "document_id": document_id,
        "chunk_id": chunk.chunk_id,
        "section": chunk.section,
        "heading": chunk.heading,
        "kind": chunk.kind,
        "category": category,
        "token_estimate": chunk.token_estimate,
        "pages": [str(p) for p in chunk.pages],
        "region_ids": json.dumps(chunk.region_ids),
        "markdown": chunk.markdown,
        "text": chunk.text,
    }


def _from_match(match: Any) -> RetrievedChunk:
    """Pinecone query match → RetrievedChunk (metadata unflattened)."""
    md = match["metadata"]
    region_ids_raw = json.loads(md.get("region_ids", "{}"))
    return RetrievedChunk(
        score=float(match["score"]),
        document_id=md["document_id"],
        chunk_id=int(md["chunk_id"]),
        section=md.get("section", ""),
        heading=md.get("heading", ""),
        markdown=md.get("markdown", ""),
        text=md.get("text", ""),
        pages=[int(p) for p in md.get("pages", [])],
        region_ids={int(k): [int(i) for i in v] for k, v in region_ids_raw.items()},
        category=md.get("category", ""),
        kind=md.get("kind", "text"),
    )


def _merge_hits(
    dense: list[RetrievedChunk], sparse: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Union of both result lists — dense order first, sparse-only appended.

    Dense and sparse scores are not comparable, so no score-based interleaving
    happens here; the caller's reranker produces the final order.
    """
    seen = {(h.document_id, h.chunk_id) for h in dense}
    return dense + [h for h in sparse if (h.document_id, h.chunk_id) not in seen]


class PineconeStore(VectorStore):
    """Vector storage on Pinecone serverless indexes (auto-created on first use).

    hybrid=True (default) maintains the sparse lexical index alongside the
    dense one; hybrid=False is dense-only, exactly the v1 behavior.
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
        """Store one dense vector per chunk (plus its sparse twin when hybrid).

        Returns the dense vector count; IDs make re-ingestion overwrite in place.
        """
        self._validate_upsert(chunks, embeddings)
        index_name = ensure_index(dimension=len(embeddings[0]))
        index = get_pinecone_client().Index(index_name)

        vectors = [
            {
                "id": _vector_id(document_id, chunk.chunk_id),
                "values": embedding,
                "metadata": _to_metadata(document_id, chunk, category),
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        t0 = time.perf_counter()
        for i in range(0, len(vectors), _UPSERT_BATCH):
            index.upsert(vectors=vectors[i : i + _UPSERT_BATCH], namespace=namespace)
        logger.info(
            "upserted %d vector(s) for doc %s in %.1fs",
            len(vectors), document_id[:12], time.perf_counter() - t0,
        )
        if self.hybrid:
            self._upsert_sparse(document_id, chunks, category, namespace)
        return len(vectors)

    def _upsert_sparse(
        self,
        document_id: str,
        chunks: list[Chunk],
        category: str,
        namespace: str,
    ) -> None:
        """Mirror the chunks into the sparse index; degrade to dense-only on failure."""
        try:
            t0 = time.perf_counter()
            sparse = embed_sparse([c.embedding_text for c in chunks], input_type="passage")
            index = get_pinecone_client().Index(ensure_sparse_index())
            vectors = [
                {
                    "id": _vector_id(document_id, chunk.chunk_id),
                    "sparse_values": {"indices": indices, "values": values},
                    "metadata": _to_metadata(document_id, chunk, category),
                }
                for chunk, (indices, values) in zip(chunks, sparse)
                if indices  # a chunk with no recognized tokens has no sparse form
            ]
            for i in range(0, len(vectors), _UPSERT_BATCH):
                index.upsert(vectors=vectors[i : i + _UPSERT_BATCH], namespace=namespace)
            logger.info(
                "upserted %d sparse vector(s) for doc %s in %.1fs",
                len(vectors), document_id[:12], time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.warning(
                "sparse upsert failed (%s: %s) — doc %s is dense-only until re-ingested",
                type(exc).__name__, exc, document_id[:12],
            )

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
        text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Nearest chunks, best first; filters are payload equality constraints.

        When hybrid and `text` is given, the sparse index is searched with the
        same top_k and its extra hits are appended after the dense results.
        """
        index = get_pinecone_client().Index(ensure_index(dimension=len(vector)))
        t0 = time.perf_counter()
        response = index.query(
            vector=vector,
            top_k=top_k,
            filter=filters,
            include_metadata=True,
            namespace=namespace,
        )
        hits = [_from_match(m) for m in response["matches"]]
        logger.info(
            "query returned %d hit(s) in %.2fs (top_k=%d, filters=%s)",
            len(hits), time.perf_counter() - t0, top_k, sorted(filters) if filters else None,
        )
        if self.hybrid and text and text.strip():
            hits = _merge_hits(hits, self._query_sparse(text, top_k, filters, namespace))
        return hits

    def _query_sparse(
        self,
        text: str,
        top_k: int,
        filters: dict[str, Any] | None,
        namespace: str,
    ) -> list[RetrievedChunk]:
        """Lexical search on the sparse index; degrades to no extra hits on failure."""
        try:
            indices, values = embed_sparse([text], input_type="query")[0]
            if not indices:
                return []
            t0 = time.perf_counter()
            index = get_pinecone_client().Index(ensure_sparse_index())
            response = index.query(
                sparse_vector={"indices": indices, "values": values},
                top_k=top_k,
                filter=filters,
                include_metadata=True,
                namespace=namespace,
            )
            hits = [_from_match(m) for m in response["matches"]]
            logger.info(
                "sparse query returned %d hit(s) in %.2fs", len(hits), time.perf_counter() - t0,
            )
            return hits
        except Exception as exc:
            logger.warning(
                "sparse query failed (%s: %s) — returning dense results only",
                type(exc).__name__, exc,
            )
            return []

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Remove the document's vectors from both indexes.

        Returns the dense count (the sparse index mirrors it 1:1, minus chunks
        that had no sparse form).
        """
        client = get_pinecone_client()
        cfg = get_pinecone_config()
        deleted = self._delete_by_prefix(cfg.index_name, document_id, namespace)
        logger.info("deleted %d vector(s) for doc %s", deleted, document_id[:12])
        if self.hybrid and client.has_index(cfg.sparse_index_name):
            n = self._delete_by_prefix(cfg.sparse_index_name, document_id, namespace)
            logger.info("deleted %d sparse vector(s) for doc %s", n, document_id[:12])
        return deleted

    @staticmethod
    def _delete_by_prefix(index_name: str, document_id: str, namespace: str) -> int:
        """List one index's vector IDs by document prefix and delete them in batches."""
        client = get_pinecone_client()
        if not client.has_index(index_name):
            return 0  # nothing was ever stored
        index = client.Index(index_name)
        deleted = 0
        for id_batch in index.list(prefix=f"{document_id}:", namespace=namespace):
            # SDK yields ListItem objects (or plain strings, version-dependent)
            ids = [getattr(item, "id", item) for item in id_batch]
            if ids:
                index.delete(ids=ids, namespace=namespace)
                deleted += len(ids)
        return deleted
