"""Storage backends for pipeline outputs.

Sub-packages / modules:
    s3        — Amazon S3 singleton client + bucket bootstrap
    artifacts — persist/load parse, classify, and split outputs on S3,
                keyed by document checksum
    base      — VectorStore contract + RetrievedChunk (works with any
                vector database via connectors)
    pinecone  — Pinecone serverless connector (index auto-created on first use)
"""
from ingestlib.storage.base import RetrievedChunk, VectorStore
from ingestlib.storage.pinecone import PineconeStore
from ingestlib.storage.s3 import ensure_bucket, get_s3_client, reset_s3_client

__all__ = [
    "get_s3_client",
    "reset_s3_client",
    "ensure_bucket",
    "VectorStore",
    "RetrievedChunk",
    "PineconeStore",
]
