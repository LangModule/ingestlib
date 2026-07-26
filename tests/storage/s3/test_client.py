"""S3 client construction and ensure_bucket error hints — no network, no mocks.

The profile test provokes a real botocore ProfileNotFound; the hint tests
exercise the pure translation on constructed botocore exceptions (the same
pattern as the bedrock hint tests).
"""
import dataclasses

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

import ingestlib.config as config_module
from ingestlib.config import get_config
from ingestlib.storage.s3.client import get_s3_client, reset_s3_client, s3_error_hint


@pytest.fixture()
def patched_aws(monkeypatch):
    """Swap the aws section, rebuilding the singleton on entry and exit."""

    def apply(**fields):
        cfg = get_config()
        patched = dataclasses.replace(
            cfg, aws=dataclasses.replace(cfg.aws, **fields)
        )
        monkeypatch.setattr(config_module, "_config", patched)
        reset_s3_client()

    yield apply
    reset_s3_client()


def test_bogus_profile_fails_loudly_not_silently(patched_aws):
    """The old behavior fell back to the default session — a typo'd profile
    must never touch a different account's bucket."""
    patched_aws(profile="definitely-not-a-real-profile")
    with pytest.raises(RuntimeError, match="available profiles"):
        get_s3_client()


def test_read_timeout_fits_small_artifacts(patched_aws):
    patched_aws(profile="")  # default chain — builds without touching ~/.aws profiles
    client = get_s3_client()
    assert client.meta.config.read_timeout == 120


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "HeadBucket")


def test_hint_403_names_global_buckets_and_the_config_key():
    hint = s3_error_hint(_client_error("403"), "taken-name")
    assert "another AWS account" in hint
    assert "s3.bucket" in hint
    assert "taken-name" in hint


def test_hint_bucket_already_exists_matches_the_create_race():
    assert "bucket names are global" in s3_error_hint(
        _client_error("BucketAlreadyExists"), "taken-name"
    )


def test_hint_expired_token_points_at_sso_login():
    assert "aws sso login" in s3_error_hint(_client_error("ExpiredToken"), "b")


def test_hint_missing_credentials_points_at_sso_login():
    assert "aws sso login" in s3_error_hint(NoCredentialsError(), "b")


def test_hint_unknown_error_returns_none_so_the_original_surfaces():
    assert s3_error_hint(_client_error("SlowDown"), "b") is None
    assert s3_error_hint(ValueError("x"), "b") is None
