"""Artifact BYTES round-trip against the MinIO container — the whole `s3` code
path with no AWS account. Opt-in via RUN_MINIO_E2E=1 (needs the minio container:
docker compose -f infra/docker-compose.yml --profile minio up -d).

Proves the endpoint-override end to end: a config of `artifact_store: s3` +
`s3.endpoint_url` (+ static creds in .env) routes S3BlobStore at MinIO, and
ensure_bucket/put/get/delete all work against it. Mirrors test_artifacts_e2e's
synthetic ParseResult; structure/metadata live in the registry, not here.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MINIO_E2E") != "1",
    reason="MinIO e2e is opt-in: set RUN_MINIO_E2E=1 with the minio container up",
)

_ENDPOINT = os.environ.get("MINIO_ENDPOINT_URL", "http://localhost:9000")
_DOC_ID = "minio-e2e-" + "0" * 54  # sentinel checksum-shaped id, safe to delete


@pytest.fixture()
def minio_artifacts(tmp_path, monkeypatch):
    """Point the config at MinIO (artifact_store: s3 + endpoint + static creds),
    rebuild the singletons, and restore the real config afterwards."""
    from ingestlib.config import reset_config

    (tmp_path / "config.yaml").write_text(
        "artifact_store: s3\n"
        "llm_provider: openai\n"
        "embedding_provider: openai\n"
        "reranker: none\n"
        f"s3:\n  bucket: ingestlib-minio-e2e\n  endpoint_url: {_ENDPOINT}\n"
    )
    (tmp_path / ".env").write_text(
        "AWS_ACCESS_KEY_ID=minioadmin\nAWS_SECRET_ACCESS_KEY=minioadmin\n"
    )
    monkeypatch.setenv("INGESTLIB_CONFIG", str(tmp_path / "config.yaml"))
    reset_config()
    yield
    reset_config()


def _synthetic_parse_result(doc_id: str = _DOC_ID):
    from ingestlib.foundations.ocr.models import BoundingBox, Region
    from ingestlib.operations.parse.models import FigureImage, PageResult, ParseResult

    region = Region(
        region_type="chart",
        bbox=BoundingBox(x=10, y=20, width=100, height=50),
        region_id=0,
        text="chart data",
        content="| a | b |",
    )
    fig = FigureImage(
        region_id=0, region_type="chart", image_bytes=b"\x89PNG-fig", caption="Fig 1"
    )
    page = PageResult(
        page_num=1,
        text="hello",
        markdown="# hello",
        regions=[region],
        figures=[fig],
        native_text="hello native",
        image_bytes=b"\x89PNG-page",
        page_width=100,
        page_height=200,
    )
    return ParseResult(
        pages=[page],
        source_path=Path("synthetic.pdf"),
        source_format="pdf",
        source_checksum=doc_id,
    )


def test_bytes_round_trip_against_minio(minio_artifacts):
    from ingestlib.storage import artifacts

    doc_id = artifacts.save_parse(_synthetic_parse_result())
    try:
        assert doc_id == _DOC_ID
        assert artifacts.read_blob(artifacts.page_image_key(doc_id, 1)) == b"\x89PNG-page"
        assert artifacts.document_markdown(doc_id) == "# hello"
        # synthetic.pdf never existed on disk, so no source blob was written
        assert artifacts.missing_blobs(doc_id, "synthetic.pdf") == ["source"]
    finally:
        removed = artifacts.delete_document(doc_id)
    assert removed >= 2  # document.md + page render + figure crop
    assert artifacts.delete_document(doc_id) == 0
