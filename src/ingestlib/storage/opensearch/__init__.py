"""OpenSearch connector — faiss k-NN + Lucene BM25 hybrid, fused with RRF.

Works against an Amazon OpenSearch Service domain (requests SigV4-sign with
the configured aws profile — no separate key) or a local server (docker).
One endpoint (OPENSEARCH_URL in .env) — the index bootstraps on first use.
"""
from ingestlib.storage.opensearch.client import (
    ensure_index,
    get_opensearch_client,
    reset_opensearch_client,
)
from ingestlib.storage.opensearch.store import OpensearchStore

__all__ = [
    "OpensearchStore",
    "get_opensearch_client",
    "reset_opensearch_client",
    "ensure_index",
]
