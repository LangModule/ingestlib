"""aws_session — a typo'd profile must fail loudly, never fall back silently.

Real boto3 against the real ~/.aws: no server, no network, no mocks.
"""
import pytest

from ingestlib.utils.aws import aws_session


def test_bogus_profile_raises_with_available_profiles():
    with pytest.raises(RuntimeError, match="available profiles"):
        aws_session("definitely-not-a-real-profile", "us-east-1")


def test_bogus_profile_error_names_the_config_key():
    with pytest.raises(RuntimeError, match="aws.profile"):
        aws_session("definitely-not-a-real-profile", "us-east-1")


def test_empty_profile_uses_default_chain():
    session = aws_session("", "us-east-1")
    assert session.region_name == "us-east-1"


def test_whitespace_profile_counts_as_empty():
    session = aws_session("   ", "us-east-1")
    assert session.region_name == "us-east-1"
