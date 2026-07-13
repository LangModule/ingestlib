"""Amazon S3 backend — singleton client accessors + bucket bootstrap."""
from ingestlib.storage.s3.client import ensure_bucket, get_s3_client, reset_s3_client

__all__ = ["get_s3_client", "reset_s3_client", "ensure_bucket"]
