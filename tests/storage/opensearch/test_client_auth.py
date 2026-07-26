"""Amazon-endpoint signing must fail loudly on a typo'd profile — no network.

Credential resolution reads the local ~/.aws only, so this runs ungated;
the bogus profile provokes a real botocore ProfileNotFound.
"""
import dataclasses

import pytest

import ingestlib.config as config_module
from ingestlib.config import get_config
from ingestlib.storage.opensearch import get_opensearch_client, reset_opensearch_client


@pytest.fixture()
def amazon_endpoint_with_bogus_profile(monkeypatch):
    cfg = get_config()
    patched = dataclasses.replace(
        cfg,
        aws=dataclasses.replace(cfg.aws, profile="definitely-not-a-real-profile"),
        opensearch=dataclasses.replace(
            cfg.opensearch,
            url="https://search-x-abc123.us-east-1.es.amazonaws.com",
        ),
    )
    monkeypatch.setattr(config_module, "_config", patched)
    reset_opensearch_client()
    yield
    reset_opensearch_client()


def test_bogus_profile_fails_loudly_before_any_request(
    amazon_endpoint_with_bogus_profile,
):
    with pytest.raises(RuntimeError, match="available profiles"):
        get_opensearch_client()
