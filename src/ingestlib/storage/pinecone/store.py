"""PineconeStore — the VectorStore contract implemented on Pinecone serverless.

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
from ingestlib.storage.pinecone.client import ensure_index, get_pinecone_client
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


class PineconeStore(VectorStore):
    """Vector storage on a Pinecone serverless index (auto-created on first use)."""

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        category: str = "",
        namespace: str = "",
    ) -> int:
        """Store one vector per chunk; IDs make re-ingestion overwrite in place."""
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
        return len(vectors)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str = "",
    ) -> list[RetrievedChunk]:
        """Nearest chunks, best first; filters are payload equality constraints."""
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
        return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """List this document's vector IDs by prefix and delete them in batches."""
        client = get_pinecone_client()
        index_name = get_pinecone_config().index_name
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
        logger.info("deleted %d vector(s) for doc %s", deleted, document_id[:12])
        return deleted
