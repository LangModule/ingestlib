"""SqliteStore — the VectorStore contract on a single local SQLite file.

Hybrid by default: the dense side is a vec0 KNN table (cosine, namespace as
the partition key, filter fields as indexed metadata columns) and the lexical
side is FTS5's native BM25 with porter stemming, the chunk breadcrumb weighted
above the body. Queries that carry the original question text run both and
fuse with Reciprocal Rank Fusion; the caller's reranker produces the final
order on top. Lexical failures degrade to dense-only with a warning.

What the embedded engine buys over the remote connectors:
  - upserts and deletes are ONE transaction across row store, vector table,
    and text index — a document is fully searchable or absent, never half
  - chunk text is stored once; FTS5 indexes it in place (external content)
    and triggers keep that index consistent with the row store
  - the namespace filter is a physical partition, not a payload match
  - deletion is plain SQL with an exact count — no listing dance

Backend quirks handled here so callers never see them:
  - FTS5 MATCH has query syntax, so a raw natural question is a syntax
    error — queries are reduced to quoted OR'd word tokens; BM25 does the
    ranking, recall stays wide
  - dense-only scores are cosine similarity; fused scores are RRF ranks —
    same convention as the other hybrid connectors
"""
import json
import re
import sqlite3
import time
from typing import Any

from sqlite_vec import serialize_float32

from ingestlib.operations.split.models import Chunk
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.sqlite.client import connection, ensure_schema, schema_exists
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

# Fields queries may filter on — each is an indexed vec0 metadata column AND
# a chunks column, so dense and lexical branches accept the same filters.
_FILTERABLE = ("document_id", "category", "section", "kind")

# bm25() weights: breadcrumb (category/section/heading words), then body.
_BM25 = "bm25(chunks_fts, 2.0, 1.0)"

_RRF_K = 60


def _breadcrumb(chunk: Chunk, category: str) -> str:
    return " ".join(part for part in (category, chunk.section, chunk.heading) if part)


def _to_payload(
    document_id: str, chunk: Chunk, category: str, namespace: str
) -> dict[str, Any]:
    """Chunk → payload dict stored as JSON (region_ids keys stringified)."""
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


def _fts_match(text: str) -> str | None:
    """Natural question → safe FTS5 query: quoted word tokens OR'd together.

    Raw text is MATCH syntax ('+360%?' is a parse error, not a search);
    None when nothing tokenizes.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _filter_sql(filters: dict[str, Any] | None, prefix: str = "") -> tuple[str, list[Any]]:
    """Equality constraints on filterable columns → SQL fragment + params."""
    unknown = set(filters or ()) - set(_FILTERABLE)
    if unknown:
        raise ValueError(
            f"unsupported filter field(s) {sorted(unknown)} — the sqlite "
            f"connector filters on {list(_FILTERABLE)}"
        )
    sql, params = "", []
    for key, value in (filters or {}).items():
        sql += f" AND {prefix}{key} = ?"
        params.append(value)
    return sql, params


def _rrf(dense: list[int], sparse: list[int]) -> list[tuple[int, float]]:
    """Fuse two rank lists — best fused score first, dense wins ties."""
    scores: dict[int, float] = {}
    for rowids in (dense, sparse):
        for rank, rowid in enumerate(rowids):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _delete_rows(conn: sqlite3.Connection, document_id: str, namespace: str) -> int:
    """Remove a document's rows from all three tables (caller owns the transaction)."""
    rowids = [
        rowid
        for (rowid,) in conn.execute(
            "SELECT rowid FROM chunks WHERE namespace = ? AND document_id = ?",
            (namespace, document_id),
        )
    ]
    for rowid in rowids:
        conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
    if rowids:
        # the delete trigger keeps chunks_fts in sync
        conn.execute(
            "DELETE FROM chunks WHERE namespace = ? AND document_id = ?",
            (namespace, document_id),
        )
    return len(rowids)


class SqliteStore(VectorStore):
    """Vector storage in a local SQLite file (schema auto-created on first use).

    hybrid=True (default) runs FTS5 BM25 next to every dense query and fuses
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
        """Store one row per chunk across all three tables in ONE transaction.

        Returns the chunk count; the document's previous rows are replaced
        atomically, so re-ingestion overwrites, never duplicates.
        """
        self._validate_upsert(chunks, embeddings)
        with connection() as conn:
            ensure_schema(conn, dimension=len(embeddings[0]))
            t0 = time.perf_counter()
            conn.execute("BEGIN IMMEDIATE")
            try:
                _delete_rows(conn, document_id, namespace)
                for chunk, embedding in zip(chunks, embeddings):
                    cursor = conn.execute(
                        "INSERT INTO chunks (document_id, chunk_id, namespace, category,"
                        " section, kind, breadcrumb, body, payload)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            document_id, chunk.chunk_id, namespace, category,
                            chunk.section, chunk.kind, _breadcrumb(chunk, category),
                            chunk.text or chunk.markdown,
                            json.dumps(_to_payload(document_id, chunk, category, namespace)),
                        ),
                    )
                    conn.execute(
                        "INSERT INTO chunks_vec (rowid, embedding, namespace, document_id,"
                        " category, section, kind) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            cursor.lastrowid, serialize_float32(embedding), namespace,
                            document_id, category, chunk.section, chunk.kind,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
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

        When hybrid and `text` is given, a BM25 branch runs next to the KNN
        and both fuse with RRF — scores are then RRF ranks, not cosine.
        """
        filter_sql, filter_params = _filter_sql(filters)
        with connection() as conn:
            ensure_schema(conn, dimension=len(vector))
            t0 = time.perf_counter()

            dense = conn.execute(
                "SELECT rowid, distance FROM chunks_vec"
                f" WHERE embedding MATCH ? AND k = ? AND namespace = ?{filter_sql}",
                [serialize_float32(vector), top_k, namespace, *filter_params],
            ).fetchall()

            sparse_rowids: list[int] = []
            if self.hybrid and text and text.strip():
                match = _fts_match(text)
                if match is not None:
                    joined_sql, joined_params = _filter_sql(filters, prefix="c.")
                    try:
                        sparse_rowids = [
                            rowid
                            for (rowid,) in conn.execute(
                                "SELECT f.rowid FROM chunks_fts f"
                                " JOIN chunks c ON c.rowid = f.rowid"
                                f" WHERE chunks_fts MATCH ? AND c.namespace = ?{joined_sql}"
                                f" ORDER BY {_BM25} LIMIT ?",
                                [match, namespace, *joined_params, top_k],
                            )
                        ]
                    except sqlite3.OperationalError as exc:
                        logger.warning("BM25 branch failed (%s) — dense-only query", exc)

            if sparse_rowids:
                ranked = _rrf([rowid for rowid, _ in dense], sparse_rowids)[:top_k]
            else:
                ranked = [(rowid, 1.0 - distance) for rowid, distance in dense]

            hits = self._load(conn, ranked)
            logger.info(
                "query returned %d hit(s) in %.2fs (top_k=%d, hybrid=%s, filters=%s)",
                len(hits), time.perf_counter() - t0, top_k, bool(sparse_rowids),
                sorted(filters) if filters else None,
            )
            return hits

    def delete_document(self, document_id: str, namespace: str = "") -> int:
        """Delete the document's rows from all three tables. Returns count removed."""
        with connection() as conn:
            if not schema_exists(conn):
                return 0  # nothing was ever stored
            conn.execute("BEGIN IMMEDIATE")
            try:
                count = _delete_rows(conn, document_id, namespace)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            logger.info("deleted %d chunk(s) for doc %s", count, document_id[:12])
            return count

    @staticmethod
    def _load(
        conn: sqlite3.Connection, ranked: list[tuple[int, float]]
    ) -> list[RetrievedChunk]:
        """Payloads for ranked (rowid, score) pairs, preserving rank order."""
        if not ranked:
            return []
        rowids = [rowid for rowid, _ in ranked]
        payloads = {
            rowid: json.loads(payload)
            for rowid, payload in conn.execute(
                f"SELECT rowid, payload FROM chunks"
                f" WHERE rowid IN ({','.join('?' * len(rowids))})",
                rowids,
            )
        }
        return [
            _from_payload(score, payloads[rowid])
            for rowid, score in ranked
            if rowid in payloads
        ]
