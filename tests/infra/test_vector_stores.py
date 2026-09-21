"""Layer A — the vector-store servers: reachable, right version, and carrying
the specific capability the connector depends on (pgvector's extension,
opensearch's knn plugin, mongodb's mongot search, qdrant/weaviate readiness +
their gRPC ports).

Opt-in via RUN_INFRA_E2E=1; each test skips if its service is unreachable.
Connection details come from the connector env vars (defaults match the compose).
"""
import json
import os
from urllib.parse import urlsplit

import pytest

from tests.infra._util import http_get, infra_e2e, pg_url_plain, reachable_http, tcp_open

pytestmark = infra_e2e


def _require_http(url: str):
    if not reachable_http(url):
        pytest.skip(f"unreachable: {url}")


def test_qdrant_ready_and_versioned():
    base = os.environ.get("QDRANT_URL", "http://localhost:6333")
    _require_http(f"{base}/readyz")
    assert http_get(f"{base}/readyz")[0] == 200
    status, body = http_get(f"{base}/")
    assert status == 200 and json.loads(body).get("version"), "no version in qdrant root"
    assert tcp_open(urlsplit(base).hostname or "localhost", 6334), "qdrant gRPC port 6334 not open"


def test_pgvector_has_extension_and_cosine_op():
    psycopg = pytest.importorskip("psycopg")
    url = pg_url_plain(os.environ.get("PGVECTOR_URL", "postgresql://postgres:pw@localhost:5432/postgres"))
    try:
        conn = psycopg.connect(url, autocommit=True)
    except Exception as exc:
        pytest.skip(f"pgvector unreachable: {exc}")
    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
        assert ext, "vector extension not installed"
        distance = conn.execute("SELECT '[1,0]'::vector <=> '[0,1]'::vector").fetchone()[0]
        assert distance == pytest.approx(1.0), "cosine operator misbehaving"


def test_opensearch_healthy_with_knn_plugin():
    base = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
    _require_http(base)
    status, body = http_get(f"{base}/_cluster/health")
    # single-node is normally yellow (no replicas); red = broken
    assert status == 200 and json.loads(body)["status"] in ("green", "yellow")
    _, plugins = http_get(f"{base}/_cat/plugins?h=component")
    assert b"knn" in (plugins or b"").lower(), "opensearch-knn plugin not loaded"


def test_weaviate_ready_live_and_versioned():
    base = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    _require_http(f"{base}/v1/.well-known/ready")
    assert http_get(f"{base}/v1/.well-known/ready")[0] == 200
    assert http_get(f"{base}/v1/.well-known/live")[0] == 200
    status, body = http_get(f"{base}/v1/meta")
    assert status == 200 and json.loads(body).get("version"), "no version in weaviate meta"
    assert tcp_open(urlsplit(base).hostname or "localhost", 50051), "weaviate gRPC port 50051 not open (the v4 client needs it)"


def test_mongodb_primary_with_search():
    pymongo = pytest.importorskip("pymongo")
    url = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/?directConnection=true")
    try:
        client = pymongo.MongoClient(url, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
    except Exception as exc:
        pytest.skip(f"mongodb unreachable: {exc}")
    try:
        assert client.admin.command("replSetGetStatus")["myState"] == 1, "not PRIMARY"
        # mongot is what makes atlas-local (vs vanilla mongo) support $vectorSearch
        list(client["infra_probe"]["coll"].aggregate([{"$listSearchIndexes": {}}]))
    finally:
        client.close()


def test_milvus_responsive():
    pymilvus = pytest.importorskip("pymilvus")
    uri = os.environ.get("MILVUS_URL", "http://localhost:19530")
    try:
        client = pymilvus.MilvusClient(uri=uri)
        client.list_collections()  # no error == server responsive
    except Exception as exc:
        pytest.skip(f"milvus unreachable: {exc}")
    finally:
        try:
            client.close()
        except Exception:
            pass
