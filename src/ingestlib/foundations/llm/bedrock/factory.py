"""Shared boto3 session, Bedrock clients, and a client-keyed model cache."""
import threading

import boto3
from botocore.config import Config

from ingestlib.config import get_aws_config, get_bedrock_config
from ingestlib.utils.aws import aws_session
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)
_lock = threading.Lock()

_session: boto3.Session | None = None
_runtime_client = None              # bedrock-runtime      (LLM inference + embeddings)
_rerank_agent_client = None         # bedrock-agent-runtime in cfg.rerank_region (for rerank)
_model_cache: dict[str, object] = {}


def _build_clients() -> None:
    global _session, _runtime_client, _rerank_agent_client

    aws = get_aws_config()
    bedrock = get_bedrock_config()
    logger.info(
        "building Bedrock clients: profile=%r region=%s rerank_region=%s",
        aws.profile, aws.region, bedrock.rerank_region,
    )

    # the shared builder fails loudly on a typo'd profile (never falls back to
    # default credentials, which could belong to a DIFFERENT account)
    _session = aws_session(aws.profile, aws.region)

    retry_cfg = Config(
        retries={"total_max_attempts": 6, "mode": "standard"},
        connect_timeout=10,
        # Long enough for a max-length generation (65535 tokens takes several
        # minutes), short enough that a wedged connection surfaces in minutes
        # instead of stalling a pipeline stage for an hour.
        read_timeout=600,
    )
    _runtime_client = _session.client(
        "bedrock-runtime", region_name=aws.region, config=retry_cfg
    )
    _rerank_agent_client = _session.client(
        "bedrock-agent-runtime", region_name=bedrock.rerank_region, config=retry_cfg
    )
    _model_cache.clear()
    logger.debug("clients built")


def _ensure() -> None:
    if _runtime_client is None:
        _build_clients()


def get_runtime_client():
    """Return the shared boto3 bedrock-runtime client (LLM + embeddings)."""
    with _lock:
        _ensure()
        return _runtime_client


def get_rerank_agent_client():
    """Return the boto3 bedrock-agent-runtime client bound to cfg.rerank_region."""
    with _lock:
        _ensure()
        return _rerank_agent_client


def reset_clients() -> None:
    """Force client recreation on the next call (e.g. after credential rotation)."""
    global _session, _runtime_client, _rerank_agent_client
    with _lock:
        logger.info("resetting Bedrock clients (next call will rebuild)")
        _session = None
        _runtime_client = None
        _rerank_agent_client = None


def bedrock_error_hint(exc: Exception) -> str | None:
    """One-sentence fix for the classic first-run failures, or None.

    Model access and expired credentials account for nearly every first
    Bedrock error, and the raw boto3 messages point users at the wrong
    place (IAM) or nowhere. Callers re-raise with the hint, keeping the
    original exception chained."""
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
    name = type(exc).__name__
    text = str(exc)

    if code == "AccessDeniedException":
        if "not authorized to perform" in text:
            return (
                "IAM denied the call — attach bedrock:InvokeModel permission "
                "for the Nova models to your profile's identity"
            )
        region = get_aws_config().region
        return (
            f"enable model access in the Bedrock console (Model access page, "
            f"region {region}) — this is separate from IAM permissions"
        )
    if code in ("ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException",
                "InvalidClientTokenId") or name in (
            "UnauthorizedSSOTokenError", "SSOTokenLoadError",
            "TokenRetrievalError", "NoCredentialsError"):
        profile = (get_aws_config().profile or "<profile>").strip() or "<profile>"
        return (
            f"AWS session expired or no credentials — run "
            f"`aws sso login --profile {profile}` (or `aws configure`)"
        )
    return None


def get_model(key: str) -> object | None:
    """Return a cached model instance, or None (reset_clients empties the cache)."""
    with _lock:
        _ensure()
        return _model_cache.get(key)


def cache_model(key: str, model: object, client: object) -> None:
    """Store a model instance — but only if `client` (the client the model was
    built around) is still the live runtime client. A reset that raced the
    build makes this a no-op instead of caching a dead-client model."""
    with _lock:
        if client is _runtime_client:
            _model_cache[key] = model
