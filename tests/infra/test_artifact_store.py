"""Layer A — the artifact-store server (MinIO): reachable on the S3 API, actually
S3-compatible for the exact usage the blob store needs (create-bucket + put/get/
list/delete under path-style addressing), and its web console live.

This is the artifact-store counterpart to the vector-store capability checks —
"is this server the thing the connector assumes it is". Opt-in via RUN_INFRA_E2E=1
and skips when minio is unreachable, so a partial `--profile` bring-up still works.
Endpoint + creds default to the compose values; override with MINIO_ENDPOINT_URL /
MINIO_ROOT_USER / MINIO_ROOT_PASSWORD.
"""
import os
import uuid

import pytest

from tests.infra._util import http_get, infra_e2e, reachable_http

pytestmark = infra_e2e

_ENDPOINT = os.environ.get("MINIO_ENDPOINT_URL", "http://localhost:9000")
_ACCESS = os.environ.get("MINIO_ROOT_USER", "minioadmin")
_SECRET = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")


def _require_minio():
    if not reachable_http(_ENDPOINT):
        pytest.skip(f"minio unreachable: {_ENDPOINT}")


def _s3_client():
    boto3 = pytest.importorskip("boto3")
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id=_ACCESS,
        aws_secret_access_key=_SECRET,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def test_minio_s3_api_answers():
    _require_minio()
    # a plain GET to the S3 root is anonymous → 403 (AccessDenied); any HTTP
    # status (not None) means the S3 API is answering, not a dead port
    assert http_get(_ENDPOINT)[0] in (200, 400, 403)


def test_minio_console_live():
    console = _ENDPOINT.replace(":9000", ":9001")
    if not reachable_http(console):
        pytest.skip(f"minio console unreachable: {console}")
    assert http_get(console)[0] == 200


def test_minio_s3_round_trip():
    """The blob store's exact usage: create a bucket, put/get/list an object, then
    delete it — S3-compatible with path-style addressing (what S3BlobStore's
    ensure_bucket + put/get/list/delete do)."""
    _require_minio()
    s3 = _s3_client()
    bucket = f"infra-probe-{uuid.uuid4().hex[:8]}"
    key = "documents/probe/document.md"
    body = b"# infra probe"

    s3.create_bucket(Bucket=bucket)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/markdown")
        assert s3.get_object(Bucket=bucket, Key=key)["Body"].read() == body
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=bucket).get("Contents", [])]
        assert key in keys, "put object not listed"
        s3.delete_object(Bucket=bucket, Key=key)
        assert s3.list_objects_v2(Bucket=bucket).get("KeyCount", 0) == 0, "object survived delete"
    finally:
        for obj in s3.list_objects_v2(Bucket=bucket).get("Contents", []):
            s3.delete_object(Bucket=bucket, Key=obj["Key"])
        s3.delete_bucket(Bucket=bucket)
