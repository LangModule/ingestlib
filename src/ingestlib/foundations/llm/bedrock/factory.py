"""Shared boto3 session, Bedrock clients, and a client-keyed model cache."""
import threading

import boto3
from botocore.config import Config
from botocore.exceptions import ProfileNotFound

from ingestlib.config import get_aws_config, get_bedrock_config
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
