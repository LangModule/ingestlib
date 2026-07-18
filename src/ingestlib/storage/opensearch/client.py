"""Process-wide singleton OpenSearch client + first-use index bootstrap.

Works against an Amazon OpenSearch Service domain (OPENSEARCH_URL in .env)
or a local server (docker run -p 9200:9200 -e discovery.type=single-node
-e DISABLE_SECURITY_PLUGIN=true opensearchproject/opensearch). Requests to
an amazonaws.com endpoint are SigV4-signed with the configured aws profile
— the same credential chain as the S3 artifact store, no separate key; any
other endpoint connects unsigned.

One index holds both retrieval signals:
    embedding         — knn_vector (HNSW cosine, faiss engine; dimension
                        from the first embedding batch)
    breadcrumb / body — text fields for BM25, breadcrumb boosted at query time
    filter cols       — namespace, document_id, category, section, kind as
                        keyword fields
    payload           — full chunk provenance, returned verbatim on hits

The index is created with zero replicas: S3 is the source of truth and
backfill rebuilds the index, so durability rides on the artifact store, not
on replica copies — and a single-node domain stays green.
"""
import re
import threading
from urllib.parse import urlparse

import boto3
from opensearchpy import OpenSearch, Urllib3AWSV4SignerAuth, Urllib3HttpConnection
from opensearchpy.exceptions import RequestError

from ingestlib.config import get_aws_config, get_opensearch_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_client: OpenSearch | None = None
_ready: dict[tuple[str, str], int] = {}     # (url, index) → dense dimension, once verified

# Amazon endpoints carry their region and service in the hostname, e.g.
# search-name-hash.us-east-1.es.amazonaws.com (aoss for serverless).
_AWS_HOST = re.compile(r"\.([a-z0-9-]+)\.(es|aoss)\.amazonaws\.com$")


def get_opensearch_client() -> OpenSearch:
    """Return the process-wide singleton OpenSearch client."""
    global _client
    with _lock:
        if _client is None:
            cfg = get_opensearch_config()
            if not cfg.url:
                raise RuntimeError(
                    "OPENSEARCH_URL is not set — add it to .env (an Amazon "
                    "OpenSearch domain endpoint, or http://localhost:9200 "
                    "for a local server)"
                )
            host = urlparse(cfg.url).hostname or ""
            aws_match = _AWS_HOST.search(host)
            auth = None
            if aws_match:
                region, service = aws_match.groups()
                aws = get_aws_config()
                credentials = boto3.Session(
                    profile_name=aws.profile or None, region_name=region
                ).get_credentials()
                if credentials is None:
                    raise RuntimeError(
                        f"no AWS credentials for profile {aws.profile!r} — "
                        f"an Amazon OpenSearch endpoint needs SigV4 signing"
                    )
                auth = Urllib3AWSV4SignerAuth(credentials, region, service)
            logger.info("building OpenSearch client: url=%s signed=%s", cfg.url, bool(auth))
            _client = OpenSearch(
                hosts=[cfg.url],
                http_auth=auth,
                connection_class=Urllib3HttpConnection,
                timeout=30,
            )
        return _client


def _index_body(dimension: int) -> dict:
    return {
        "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"},
                },
                "breadcrumb": {"type": "text"},
                "body": {"type": "text"},
                "namespace": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "category": {"type": "keyword"},
                "section": {"type": "keyword"},
                "kind": {"type": "keyword"},
                "payload": {"type": "object", "enabled": False},  # stored, never indexed
            }
        },
    }


def ensure_index(dimension: int) -> str:
    """Create the configured index on first use; verify it afterwards.

    The dense dimension comes from the first embedding batch, so the mapping
    always matches what is actually being stored — later calls with a
    different dimension fail loudly. Returns the index name.
    """
    cfg = get_opensearch_config()
    key = (cfg.url, cfg.index_name)
    client = get_opensearch_client()
    with _lock:
        known = _ready.get(key)
        if known is not None:
            if known != dimension:
                raise ValueError(
                    f"OpenSearch index {cfg.index_name!r} stores {known}-dim "
                    f"embeddings, got {dimension}-dim — use a matching embedding "
                    f"dimension or a different index_name"
                )
            return cfg.index_name

        if not client.indices.exists(index=cfg.index_name):
            logger.info(
                "creating OpenSearch index %r (dense dim=%d cosine + BM25 text) — first use",
                cfg.index_name, dimension,
            )
            try:
                client.indices.create(index=cfg.index_name, body=_index_body(dimension))
            except RequestError as exc:
                # lost a creation race with another process — fine if it exists now
                if exc.error != "resource_already_exists_exception":
                    raise
        else:
            properties = client.indices.get_mapping(index=cfg.index_name)[
                cfg.index_name
            ]["mappings"].get("properties", {})
            embedding = properties.get("embedding")
            if embedding is None or embedding.get("type") != "knn_vector":
                raise ValueError(
                    f"OpenSearch index {cfg.index_name!r} exists but has no knn_vector "
                    f"'embedding' field — it is not ingestlib's; configure a different "
                    f"index_name"
                )
            existing = int(embedding["dimension"])
            if existing != dimension:
                raise ValueError(
                    f"OpenSearch index {cfg.index_name!r} stores {existing}-dim "
                    f"embeddings, got {dimension}-dim — use a matching embedding "
                    f"dimension or a different index_name"
                )

        _ready[key] = dimension
        return cfg.index_name


def reset_opensearch_client() -> None:
    """Force client recreation on the next call (e.g. after endpoint change)."""
    global _client
    with _lock:
        if _client is not None:
            _client.close()
        _client = None
        _ready.clear()
