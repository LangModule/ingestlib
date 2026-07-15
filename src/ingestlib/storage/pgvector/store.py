"""PgvectorStore — the VectorStore contract on a Postgres table with pgvector.

Hybrid by default: the dense side is an HNSW cosine index over the pgvector
column, and the lexical side is Postgres full-text search over a GENERATED
weighted tsvector (breadcrumb 'A' over body 'B', english stemming). Queries
that carry the original question text run both and fuse with Reciprocal Rank
Fusion; the caller's reranker produces the final order on top. Lexical
failures degrade to dense-only with a warning.

The lexical ranking is ts_rank_cd, not true BM25 (stock Postgres has no IDF)
— acceptable here because RRF only consumes the *ordering* and the reranker
is the final arbiter; the eval harness watches whether that costs quality.

Like sqlite, writes are ONE transaction: a document is fully searchable or
absent. Re-ingestion deletes the document's rows first, then inserts — plain
ON CONFLICT upserts would orphan stale chunk_ids when a re-parse yields
fewer chunks.

Backend quirks handled here so callers never see them:
  - to_tsquery has query syntax, so a raw natural question is a syntax
    error — queries are reduced to OR'd word tokens ('a | b | c')
  - dense-only scores are cosine similarity (1 - <=> distance); fused
    scores are RRF ranks — same convention as the other hybrid connectors
"""
import json
import re
import time
from typing import Any

import psycopg
from pgvector import Vector

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.pgvector.client import (
    connect,
    ensure_schema,
    schema_exists,
    table_name,
)
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Fields queries may filter on — btree-backed columns on the table, valid in
# both the KNN WHERE and the lexical WHERE.
_FILTERABLE = ("document_id", "category", "section", "kind")

_RRF_K = 60


def _breadcrumb(chunk: Chunk, category: str) -> str:
    return " ".join(part for part in (category, chunk.section, chunk.heading) if part)


def _to_payload(
    document_id: str, chunk: Chunk, category: str, namespace: str
) -> dict[str, Any]:
    """Chunk → payload dict stored as JSONB (region_ids keys stringified)."""
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


def _ts_query(text: str) -> str | None:
    """Natural question → safe to_tsquery input: word tokens OR'd together.

    Raw text is tsquery syntax ('+360%?' is a parse error, not a search);
    None when nothing tokenizes.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return None
    return " | ".join(tokens)


def _filter_sql(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Equality constraints on filterable columns → SQL fragment + params."""
    unknown = set(filters or ()) - set(_FILTERABLE)
    if unknown:
        raise ValueError(
            f"unsupported filter field(s) {sorted(unknown)} — the pgvector "
            f"connector filters on {list(_FILTERABLE)}"
        )
    sql, params = "", []
    for key, value in (filters or {}).items():
        sql += f" AND {key} = %s"
        params.append(value)
    return sql, params


def _rrf(dense: list[int], sparse: list[int]) -> list[tuple[int, float]]:
    """Fuse two rank lists — best fused score first, dense wins ties."""
    scores: dict[int, float] = {}
    for row_ids in (dense, sparse):
        for rank, row_id in enumerate(row_ids):
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class PgvectorStore(VectorStore):
    """Vector storage in a Postgres table (extension + schema auto-managed).

    hybrid=True (default) runs full-text search next to every dense query
    and fuses both signals; hybrid=False is dense-only.
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
        """Replace the document's rows in ONE transaction.

        Returns the chunk count; delete-then-insert guarantees a re-parsed
        document with fewer chunks leaves no orphans behind.
        """
        self._validate_upsert(chunks, embeddings)
        with connect() as conn:
            ensure_schema(conn, dimension=len(embeddings[0]))
            table = table_name()
            t0 = time.perf_counter()
            with conn.transaction():
                conn.execute(
                    f"DELETE FROM {table} WHERE namespace = %s AND document_id = %s",
                    (namespace, document_id),
                )
                with conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO {table} (document_id, chunk_id, namespace,"
                        f" category, section, kind, breadcrumb, body, payload, embedding)"
                        f" VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        [
                            (
                                document_id, chunk.chunk_id, namespace, category,
                                chunk.section, chunk.kind, _breadcrumb(chunk, category),
                                chunk.text or chunk.markdown,
                                json.dumps(_to_payload(document_id, chunk, category, namespace)),
                                Vector(embedding),
                            )
                            for chunk, embedding in zip(chunks, embeddings)
                        ],
                    )
            logger.info(
                "upserted %d chunk(s) for doc %s in %.1fs (one transaction)",
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

        When hybrid and `text` is given, a full-text branch runs next to the
        KNN and both fuse with RRF — scores are then RRF ranks, not cosine.
        """
        filter_sql, filter_params = _filter_sql(filters)
        with connect() as conn:
            ensure_schema(conn, dimension=len(vector))
            table = table_name()
            t0 = time.perf_counter()

            dense = conn.execute(
                f"SELECT id, payload, embedding <=> %s AS distance FROM {table}"
                f" WHERE namespace = %s{filter_sql} ORDER BY distance LIMIT %s",
                [Vector(vector), namespace, *filter_params, top_k],
            ).fetchall()

            sparse: list[tuple] = []
            if self.hybrid and text and text.strip():
                ts = _ts_query(text)
                if ts is not None:
                    try:
                        sparse = conn.execute(
                            f"SELECT id, payload FROM {table},"
                            f" to_tsquery('english', %s) q"
                            f" WHERE fts @@ q AND namespace = %s{filter_sql}"
                            f" ORDER BY ts_rank_cd(fts, q) DESC LIMIT %s",
                            [ts, namespace, *filter_params, top_k],
                        ).fetchall()
                    except psycopg.Error as exc:
                        logger.warning(
                            "full-text branch failed (%s) — dense-only query", exc
                        )

            payloads = {row[0]: row[1] for row in dense}
            payloads.update({row[0]: row[1] for row in sparse})
            if sparse:
                ranked = _rrf(
                    [row[0] for row in dense], [row[0] for row in sparse]
                )[:top_k]
            else:
                ranked = [(row[0], 1.0 - float(row[2])) for row in dense]

            hits = [_from_payload(score, payloads[row_id]) for row_id, score in ranked]
            logger.info(
                "query returned %d hit(s) in %.2fs (top_k=%d, hybrid=%s, filters=%s)",
                len(hits), time.perf_counter() - t0, top_k, bool(sparse),
                sorted(filters) if filters else None,
            )
            return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's rows. Returns the exact count removed."""
        with connect() as conn:
            if not schema_exists(conn):
                return 0  # nothing was ever stored
            cursor = conn.execute(
                f"DELETE FROM {table_name()} WHERE namespace = %s AND document_id = %s",
                (namespace, document_id),
            )
            count = cursor.rowcount
            logger.info("deleted %d chunk(s) for doc %s", count, document_id[:12])
            return count
