"""Loads config.yaml at project root plus .env for secrets (JINA_API_KEY, etc.).

AWS credentials resolve via the profile field against ~/.aws/credentials.
Other provider API keys come from environment variables loaded from .env.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AWSConfig:
    profile: str
    region: str
    account_id: str


@dataclass(frozen=True)
class BedrockConfig:
    llm_model_id: str
    embedding_model_id: str
    rerank_model_id: str
    rerank_region: str          # separate region for rerank (amazon.rerank-v1:0 not in us-east-1)


@dataclass(frozen=True)
class JinaConfig:
    api_key: str                # from JINA_API_KEY env var
    base_url: str
    rerank_model_id: str


@dataclass(frozen=True)
class PaddleVLConfig:
    backend: str                # mlx-vlm-server (Apple Silicon) | vllm-server (NVIDIA)
    server_url: str             # VLM inference server URL
    api_model_name: str


@dataclass(frozen=True)
class S3Config:
    bucket: str                 # globally unique bucket name for pipeline artifacts


@dataclass(frozen=True)
class PineconeConfig:
    api_key: str                # from PINECONE_API_KEY env var
    index_name: str
    sparse_index_name: str      # lexical half of hybrid search (created on first upsert)
    sparse_model_id: str        # Pinecone-hosted sparse embedding model (no corpus state)
    cloud: str                  # serverless cloud provider (aws | gcp | azure)
    region: str


@dataclass(frozen=True)
class QdrantConfig:
    api_key: str                # from QDRANT_API_KEY env var ("" for a local/unsecured server)
    url: str                    # from QDRANT_URL env var (e.g. http://localhost:6333 or a cloud URL)
    collection_name: str


@dataclass(frozen=True)
class IngestConfig:
    vector_store: str           # which connector the services default to (pinecone | qdrant)
    aws: AWSConfig
    bedrock: BedrockConfig
    jina: JinaConfig
    paddle_vl: PaddleVLConfig
    s3: S3Config
    pinecone: PineconeConfig
    qdrant: QdrantConfig


def _load_config() -> IngestConfig:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_PATH}")

    with open(_CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)

    aws_data = data["aws"]
    aws_config = AWSConfig(
        profile=aws_data["profile"],
        region=aws_data["region"],
        account_id=str(aws_data["account_id"]),
    )

    bedrock_data = data["bedrock"]
    bedrock_config = BedrockConfig(
        llm_model_id=bedrock_data["llm_model_id"],
        embedding_model_id=bedrock_data["embedding_model_id"],
        rerank_model_id=bedrock_data["rerank_model_id"],
        rerank_region=bedrock_data["rerank_region"],
    )

    jina_data = data["jina"]
    jina_config = JinaConfig(
        api_key=os.environ.get("JINA_API_KEY", ""),
        base_url=jina_data.get("base_url", "https://api.jina.ai/v1"),
        rerank_model_id=jina_data["rerank_model_id"],
    )

    paddle_vl_data = data["paddle_vl"]
    paddle_vl_config = PaddleVLConfig(
        backend=paddle_vl_data.get("backend", "mlx-vlm-server"),
        server_url=paddle_vl_data.get("server_url", "http://localhost:8111/"),
        api_model_name=paddle_vl_data["api_model_name"],
    )

    s3_data = data.get("s3", {})
    s3_config = S3Config(
        bucket=s3_data.get("bucket", f"ingestlib-{aws_config.account_id}"),
    )

    pinecone_data = data.get("pinecone", {})
    index_name = pinecone_data.get("index_name", "ingestlib")
    pinecone_config = PineconeConfig(
        api_key=os.environ.get("PINECONE_API_KEY", ""),
        index_name=index_name,
        sparse_index_name=pinecone_data.get("sparse_index_name", f"{index_name}-sparse"),
        sparse_model_id=pinecone_data.get("sparse_model_id", "pinecone-sparse-english-v0"),
        cloud=pinecone_data.get("cloud", "aws"),
        region=pinecone_data.get("region", "us-east-1"),
    )

    qdrant_data = data.get("qdrant", {})
    qdrant_config = QdrantConfig(
        api_key=os.environ.get("QDRANT_API_KEY", ""),
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection_name=qdrant_data.get("collection_name", "ingestlib"),
    )

    return IngestConfig(
        vector_store=data.get("vector_store", "pinecone"),
        aws=aws_config,
        bedrock=bedrock_config,
        jina=jina_config,
        paddle_vl=paddle_vl_config,
        s3=s3_config,
        pinecone=pinecone_config,
        qdrant=qdrant_config,
    )


_config = _load_config()


def get_config() -> IngestConfig:
    """Full typed configuration loaded from config.yaml + .env."""
    return _config


def get_aws_config() -> AWSConfig:
    """AWS profile/region/account settings."""
    return _config.aws


def get_bedrock_config() -> BedrockConfig:
    """Bedrock model IDs and rerank region."""
    return _config.bedrock


def get_jina_config() -> JinaConfig:
    """Jina reranker endpoint and API key."""
    return _config.jina


def get_paddle_vl_config() -> PaddleVLConfig:
    """PaddleOCR-VL inference server settings."""
    return _config.paddle_vl


def get_s3_config() -> S3Config:
    """S3 artifact bucket settings."""
    return _config.s3


def get_pinecone_config() -> PineconeConfig:
    """Pinecone index settings and API key."""
    return _config.pinecone


def get_qdrant_config() -> QdrantConfig:
    """Qdrant endpoint, API key, and collection settings."""
    return _config.qdrant
