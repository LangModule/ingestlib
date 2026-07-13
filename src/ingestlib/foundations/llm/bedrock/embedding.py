"""Nova 2 multimodal embeddings — text and image embed primitives (sync and async)."""
import asyncio
import base64
import json
import time
from typing import Any, Literal

from ingestlib.config import get_bedrock_config
from ingestlib.foundations.llm.bedrock.factory import get_runtime_client
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

EmbeddingPurpose = Literal["GENERIC_INDEX", "GENERIC_RETRIEVAL", "DOCUMENT_RETRIEVAL"]
EmbeddingDimension = Literal[256, 384, 1024, 3072]
ImageFormat = Literal["jpeg", "png", "webp", "gif"]
ImageDetailLevel = Literal["STANDARD_IMAGE", "DOCUMENT_IMAGE"]

DEFAULT_DIMENSION: EmbeddingDimension = 1024
SUPPORTED_DIMENSIONS: tuple[int, ...] = (256, 384, 1024, 3072)


def _validate_dimension(dimension: int) -> None:
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(
            f"dimension must be one of {SUPPORTED_DIMENSIONS}, got {dimension}"
        )


def _invoke(body: dict[str, Any]) -> dict[str, Any]:
    client = get_runtime_client()
    cfg = get_bedrock_config()
    response = client.invoke_model(
        modelId=cfg.embedding_model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())


def _embed(
    *,
    content: dict[str, Any],
    purpose: EmbeddingPurpose,
    dimension: EmbeddingDimension,
) -> list[float]:
    _validate_dimension(dimension)
    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": purpose,
            "embeddingDimension": dimension,
            **content,
        },
    }
    return _invoke(body)["embeddings"][0]["embedding"]


def embed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Embed text → vector of `dimension` floats (truncates overlong input at the end)."""
    logger.info(
        "embed_text: purpose=%s dim=%d input_len=%d", purpose, dimension, len(text),
    )
    t0 = time.perf_counter()
    result = _embed(
        content={"text": {"truncationMode": "END", "value": text}},
        purpose=purpose,
        dimension=dimension,
    )
    logger.info("embed_text done: %.2fs returned_dim=%d", time.perf_counter() - t0, len(result))
    return result


def embed_image(
    data: bytes,
    format: ImageFormat,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
    detail_level: ImageDetailLevel = "STANDARD_IMAGE",
) -> list[float]:
    """Embed an image → vector of `dimension` floats. DOCUMENT_IMAGE detail for doc pages."""
    logger.info(
        "embed_image: purpose=%s dim=%d format=%s detail_level=%s size=%d bytes",
        purpose, dimension, format, detail_level, len(data),
    )
    t0 = time.perf_counter()
    result = _embed(
        content={
            "image": {
                "format": format,
                "detailLevel": detail_level,
                "source": {"bytes": base64.b64encode(data).decode()},
            }
        },
        purpose=purpose,
        dimension=dimension,
    )
    logger.info("embed_image done: %.2fs returned_dim=%d", time.perf_counter() - t0, len(result))
    return result


async def aembed_text(
    text: str,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
) -> list[float]:
    """Async embed_text() — runs the sync Bedrock client in a worker thread."""
    return await asyncio.to_thread(embed_text, text, purpose, dimension)


async def aembed_image(
    data: bytes,
    format: ImageFormat,
    purpose: EmbeddingPurpose = "GENERIC_INDEX",
    dimension: EmbeddingDimension = DEFAULT_DIMENSION,
    detail_level: ImageDetailLevel = "STANDARD_IMAGE",
) -> list[float]:
    """Async embed_image() — runs the sync Bedrock client in a worker thread."""
    return await asyncio.to_thread(
        embed_image, data, format, purpose, dimension, detail_level
    )
