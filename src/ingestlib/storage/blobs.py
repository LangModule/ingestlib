"""Blob backends for the artifact store — Amazon S3 or the local filesystem.

Selected by config.yaml's `artifact_store` key. Both backends speak the same
key layout (documents/{doc_id}/...), so a corpus moves between them with a
plain copy. The local backend writes real directories under artifacts.path —
relative paths anchor beside the discovered config.yaml, so a wizard-managed
machine gets ~/.ingestlib/artifacts and a per-project config gets a folder
next to it. Zero cloud, browsable in a file manager.
"""
import json
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_lock = threading.Lock()
_backend: "BlobStore | None" = None


class BlobStore(ABC):
    """The six operations the artifact store needs from its storage backend."""

    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Write bytes at `key`, creating any missing hierarchy."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Read bytes at `key`; raises when the key does not exist."""

    @abstractmethod
    def get_or_none(self, key: str) -> bytes | None:
        """Read bytes at `key` — None ONLY for a genuinely missing key.

        Every other failure (throttle, network, permissions) propagates:
        treating a transient error as absence would silently rewrite
        metadata or serve ghost registry entries.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True when an object lives at exactly `key`."""

    @abstractmethod
    def list_top_dirs(self, prefix: str) -> list[str]:
        """Immediate child directory names under `prefix` (the doc_id registry)."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Remove everything under `prefix`. Returns the object count removed."""

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.put(key, json.dumps(payload, ensure_ascii=False).encode(), "application/json")


class S3BlobStore(BlobStore):
    """Artifacts in the configured S3 bucket (created on first use)."""

    def put(self, key: str, data: bytes, content_type: str) -> None:
        from ingestlib.storage.s3.client import ensure_bucket, get_s3_client

        get_s3_client().put_object(
            Bucket=ensure_bucket(), Key=key, Body=data, ContentType=content_type
        )

    def get(self, key: str) -> bytes:
        from ingestlib.storage.s3.client import ensure_bucket, get_s3_client

        response = get_s3_client().get_object(Bucket=ensure_bucket(), Key=key)
        return response["Body"].read()

    def get_or_none(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            return self.get(key)
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def exists(self, key: str) -> bool:
        from ingestlib.storage.s3.client import ensure_bucket, get_s3_client

        response = get_s3_client().list_objects_v2(
            Bucket=ensure_bucket(), Prefix=key, MaxKeys=1
        )
        return response.get("KeyCount", 0) > 0

    def list_top_dirs(self, prefix: str) -> list[str]:
        from ingestlib.storage.s3.client import ensure_bucket, get_s3_client

        client = get_s3_client()
        names: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=ensure_bucket(), Prefix=f"{prefix}/", Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                names.append(cp["Prefix"].rstrip("/").split("/")[-1])
        return names

    def delete_prefix(self, prefix: str) -> int:
        from ingestlib.storage.s3.client import ensure_bucket, get_s3_client

        client = get_s3_client()
        bucket = ensure_bucket()
        deleted = 0
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if not keys:
                continue
            response = client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            errors = response.get("Errors", [])
            if errors:  # partial failure must not be reported as success
                raise RuntimeError(
                    f"delete under {prefix!r}: {len(errors)} object(s) failed, "
                    f"first: {errors[0].get('Key')} ({errors[0].get('Message')})"
                )
            deleted += len(keys)
        return deleted


class LocalBlobStore(BlobStore):
    """Artifacts as plain files under artifacts.path — no cloud, no server.

    Writes go through a temp file + atomic rename, so a crash mid-write can
    never leave a truncated result.json behind.
    """

    def __init__(self) -> None:
        from ingestlib.config import get_artifacts_config

        self.root = get_artifacts_config().path

    def _path(self, key: str) -> Path:
        return self.root / key

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_or_none(self, key: str) -> bytes | None:
        try:
            return self.get(key)
        except FileNotFoundError:
            return None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_top_dirs(self, prefix: str) -> list[str]:
        directory = self._path(prefix)
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_dir())

    def delete_prefix(self, prefix: str) -> int:
        directory = self._path(prefix.rstrip("/"))
        if not directory.is_dir():
            return 0
        count = sum(1 for p in directory.rglob("*") if p.is_file())
        shutil.rmtree(directory)
        return count


_BACKENDS: dict[str, type[BlobStore]] = {
    "s3": S3BlobStore,
    "local": LocalBlobStore,
}


def get_blob_store() -> BlobStore:
    """The backend selected by config.yaml's `artifact_store` key (cached)."""
    global _backend
    with _lock:
        if _backend is None:
            from ingestlib.config import get_config

            name = get_config().artifact_store
            if name not in _BACKENDS:
                raise ValueError(
                    f"unknown artifact_store {name!r} in config.yaml — "
                    f"choose one of {sorted(_BACKENDS)}"
                )
            logger.info("artifact store backend: %s", name)
            _backend = _BACKENDS[name]()
        return _backend


def reset_blob_store() -> None:
    """Forget the cached backend so the next call re-reads the config."""
    global _backend
    with _lock:
        _backend = None
