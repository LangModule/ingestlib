"""Client factory behavior — pure, always run.

A typo'd aws.profile must fail loudly with the available names, never fall
back to the default credential chain (which could be a different account).
"""
import pytest

import ingestlib.foundations.llm.bedrock.factory as factory


def test_missing_profile_raises_with_available_names(monkeypatch):
    from ingestlib.config import AWSConfig, BedrockConfig

    monkeypatch.setattr(
        factory, "get_aws_config",
        lambda: AWSConfig(profile="no-such-profile-xyz", region="us-east-1", account_id="1"),
    )
    monkeypatch.setattr(
        factory, "get_bedrock_config",
        lambda: BedrockConfig(
            llm_model_id="m", embedding_model_id="e",
            rerank_model_id="r", rerank_region="us-west-2",
        ),
    )
    factory.reset_clients()
    try:
        with pytest.raises(RuntimeError, match="no-such-profile-xyz"):
            factory.get_runtime_client()
    finally:
        factory.reset_clients()


def _client_error(code: str, message: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


@pytest.fixture()
def pinned_aws(monkeypatch):
    from ingestlib.config import AWSConfig

    monkeypatch.setattr(
        factory, "get_aws_config",
        lambda: AWSConfig(profile="acme-prod", region="us-east-1", account_id="1"),
    )


def test_hint_model_access_names_the_console_and_region(pinned_aws):
    exc = _client_error(
        "AccessDeniedException",
        "You don't have access to the model with the specified model ID.",
    )
    hint = factory.bedrock_error_hint(exc)
    assert "Model access" in hint and "us-east-1" in hint


def test_hint_iam_denial_names_iam_not_the_console(pinned_aws):
    exc = _client_error(
        "AccessDeniedException",
        "User: arn:aws:iam::1:user/x is not authorized to perform: bedrock:InvokeModel",
    )
    hint = factory.bedrock_error_hint(exc)
    assert "IAM" in hint and "bedrock:InvokeModel" in hint


def test_hint_expired_token_names_the_profile(pinned_aws):
    exc = _client_error("ExpiredTokenException", "The security token included is expired")
    hint = factory.bedrock_error_hint(exc)
    assert "aws sso login" in hint and "acme-prod" in hint


def test_hint_missing_credentials(pinned_aws):
    from botocore.exceptions import NoCredentialsError

    assert "credentials" in factory.bedrock_error_hint(NoCredentialsError()).lower()


def test_hint_unknown_error_returns_none(pinned_aws):
    assert factory.bedrock_error_hint(_client_error("ValidationException", "bad input")) is None
    assert factory.bedrock_error_hint(ValueError("nope")) is None
