"""Milvus connector — dense ANN + server-side BM25 hybrid, fused with RRF.

Works against a local standalone server (docker) or Zilliz Cloud. One
endpoint (MILVUS_URL, plus MILVUS_TOKEN for secured deployments) — the
collection, indexes, and BM25 function bootstrap on first use.
"""
from ingestlib.storage.milvus.store import MilvusStore

__all__ = ["MilvusStore"]
