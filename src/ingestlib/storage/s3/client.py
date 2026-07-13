"""Process-wide singleton boto3 S3 client + first-time bucket bootstrap."""
import threading

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ProfileNotFound

from ingestlib.config import get_aws_config, get_s3_config
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_session: boto3.Session | None = None
_s3_client = None
_bucket_ready = False


def _build_client() -> None:
    global _session, _s3_client

    aws = get_aws_config()
    profile = (aws.profile or "").strip()
    try:
        _session = (
            boto3.Session(profile_name=profile, region_name=aws.region)
            if profile
            else boto3.Session(region_name=aws.region)
        )
    except ProfileNotFound:
        logger.warning("profile %r not found, falling back to default session", profile)
        _session = boto3.Session(region_name=aws.region)

    retry_cfg = Config(
        retries={"total_max_attempts": 6, "mode": "standard"},
        connect_timeout=10,
        read_timeout=3600,
    )
    _s3_client = _session.client("s3", region_name=aws.region, config=retry_cfg)


def get_s3_client():
    """Return the process-wide singleton boto3 S3 client."""
    with _lock:
        if _s3_client is None:
            _build_client()
        return _s3_client


def ensure_bucket() -> str:
    """Create the artifact bucket on first use; no-op once it exists.

    Returns the bucket name. Handles the us-east-1 API quirk (CreateBucket
    rejects a LocationConstraint there) and races where the bucket was just
    created by another process.
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
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise  # 403 = exists but not ours (names are global) — surface it

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
