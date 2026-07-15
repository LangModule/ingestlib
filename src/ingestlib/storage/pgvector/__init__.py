"""pgvector connector — vector storage in the Postgres you already run.

Dense KNN via the pgvector extension (HNSW cosine), lexical via built-in
full-text search (generated weighted tsvector), fused with RRF. One
connection URL (PGVECTOR_URL in .env) — the extension is verified/enabled
and the table bootstraps on first use.
"""
from ingestlib.storage.pgvector.store import PgvectorStore

__all__ = ["PgvectorStore"]
