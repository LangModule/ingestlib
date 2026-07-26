"""Process-wide singleton boto3 S3 client + first-time bucket bootstrap."""
import threading

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ingestlib.config import get_aws_config, get_s3_config
from ingestlib.utils.aws import aws_session
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_session: boto3.Session | None = None
_s3_client = None
_bucket_ready = False


def _build_client() -> None:
    global _session, _s3_client

    aws = get_aws_config()
    _session = aws_session(aws.profile, aws.region)

    retry_cfg = Config(
        retries={"total_max_attempts": 6, "mode": "standard"},
        connect_timeout=10,
        # Artifacts are small JSON files and page PNGs — a stuck read should
        # surface in minutes, not stall a pipeline stage for an hour.
        read_timeout=120,
    )
    _s3_client = _session.client("s3", region_name=aws.region, config=retry_cfg)


def get_s3_client():
    """Return the process-wide singleton boto3 S3 client."""
    with _lock:
        if _s3_client is None:
            _build_client()
        return _s3_client


def s3_error_hint(exc: Exception, bucket: str) -> str | None:
    """One-sentence fix for the classic ensure_bucket failures, else None."""
    name = type(exc).__name__
    code = ""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")

    if code in ("403", "Forbidden", "AccessDenied", "BucketAlreadyExists"):
        # S3 bucket names are global across ALL accounts — the usual cause of
        # a 403 on a bucket you never created is someone else owning the name.
        return (
            f"S3 bucket {bucket!r} already exists in another AWS account "
            f"(bucket names are global) or your credentials can't access it — "
            f"pick a unique `s3.bucket` in config.yaml"
        )
    if code in ("ExpiredToken", "ExpiredTokenException", "InvalidAccessKeyId",
                "InvalidClientTokenId") or name in (
            "NoCredentialsError", "UnauthorizedSSOTokenError",
            "SSOTokenLoadError", "TokenRetrievalError"):
        profile = (get_aws_config().profile or "<profile>").strip() or "<profile>"
        return (
            f"AWS session expired or no credentials — run "
            f"`aws sso login --profile {profile}` (or `aws configure`)"
        )
    return None


def ensure_bucket() -> str:
    """Create the artifact bucket on first use; no-op once it exists.

    Returns the bucket name. Handles the us-east-1 API quirk (CreateBucket
    rejects a LocationConstraint there) and races where the bucket was just
    created by another process. Classic failures — name taken by another
    account, expired credentials — re-raise with a one-sentence fix.
    """
    global _bucket_ready
    bucket = get_s3_config().bucket
    if _bucket_ready:
        return bucket

    client = get_s3_client()
    region = get_aws_config().region
    try:
        client.head_bucket(Bucket=bucket)
        _bucket_ready = True
        return bucket
    except Exception as exc:
        code = exc.response["Error"]["Code"] if isinstance(exc, ClientError) else ""
        if code not in ("404", "NoSuchBucket"):
            hint = s3_error_hint(exc, bucket)
            if hint:
                raise RuntimeError(hint) from exc
            raise

    logger.info("creating S3 bucket %r in %s (first use)", bucket, region)
    try:
        if region == "us-east-1":
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou",):
            hint = s3_error_hint(exc, bucket)
            if hint:
                raise RuntimeError(hint) from exc
            raise
    client.get_waiter("bucket_exists").wait(Bucket=bucket)
    logger.info("S3 bucket %r ready", bucket)
    _bucket_ready = True
    return bucket


def reset_s3_client() -> None:
    """Force client recreation on the next call (e.g. after credential rotation)."""
    global _session, _s3_client, _bucket_ready
    with _lock:
        _session = None
        _s3_client = None
        _bucket_ready = False
