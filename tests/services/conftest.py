"""Shared lifecycle fixtures for the services suites — always run, no gates.

The whole lifecycle surface is testable with zero servers: artifacts on the
local filesystem backend, vectors in sqlite, synthetic 8-dim embeddings.
`stack` wires both stores under tmp_path; `make_document` fabricates a fully
"ingested" document (parse + split artifacts, manifest, vectors) without any
pipeline or LLM call.
"""
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

import ingestlib.config as config_module
from ingestlib.config import ArtifactsConfig, SqliteConfig, get_config
from ingestlib.operations.split.models import Chunk, Section, SplitResult
from ingestlib.storage import SqliteStore, artifacts
from ingestlib.storage.blobs import reset_blob_store

DIM = 8


def vec(*values: float) -> list[float]:
    return list(values) + [0.0] * (DIM - len(values))


def synthetic_parse_result(doc_id: str, source: Path):
    from ingestlib.foundations.ocr.models import BoundingBox, Region
    from ingestlib.operations.parse.models import PageResult, ParseResult

    region = Region(
        region_type="text",
        bbox=BoundingBox(x=10, y=20, width=100, height=50),
        region_id=0,
        text="hello",
        content="hello",
    )
    page = PageResult(
        page_num=1, text="hello", markdown="# hello", regions=[region],
        figures=[], native_text="hello", image_bytes=b"\x89PNG-page",
        page_width=100, page_height=200,
    )
    return ParseResult(
        pages=[page], source_path=source, source_format="pdf",
        source_checksum=doc_id,
    )


@pytest.fixture()
def stack(tmp_path, monkeypatch):
    """Local artifacts + sqlite vectors under tmp_path, plus a corpus dir."""
    cfg = dataclasses.replace(
        get_config(),
        artifact_store="local",
        artifacts=ArtifactsConfig(path=tmp_path / "artifacts"),
        vector_store="sqlite",
        sqlite=SqliteConfig(path=tmp_path / "vectors.db"),
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    reset_blob_store()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    yield SimpleNamespace(root=tmp_path, corpus=corpus, store=SqliteStore())
    reset_blob_store()


@pytest.fixture()
def pipeline(stack, monkeypatch):
    """Stub the four model boundaries at the ingestor's seams — artifact
    saves and vector upserts stay REAL (local files + sqlite), so the
    registry, manifests, and deletion behave exactly as in production.
    The fake parse computes the file's true sha256, matching aingest's."""
    import importlib

    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.utils.files import sha256_of_file

    ingestor = importlib.import_module("ingestlib.services.ingest.ingestor")

    async def fake_aparse(path, *, dpi=200):
        path = Path(path)
        return synthetic_parse_result(sha256_of_file(path), path)

    async def fake_aclassify(source, categories=None, *, target_pages=None, max_pages=None):
        return ClassifyResult(category="report", confidence=0.9)

    async def fake_asplit(source, *, category=None, max_chunk_tokens=768,
                          vocabulary=None, unmatched=None):
        chunk = Chunk(
            chunk_id=0, section="body", heading="h", text="hello",
            markdown="hello", embedding_text="[doc › body › h]\n\nhello",
            pages=[1], region_ids={1: [0]},
        )
        return SplitResult(
            sections=[Section(name="body", pages=[1], chunks=[chunk])],
            pages_used=1,
        )

    async def fake_embed(text, purpose="GENERIC_INDEX", dimension=1024):
        return vec(1.0)

    monkeypatch.setattr(ingestor, "aparse", fake_aparse)
    monkeypatch.setattr(ingestor, "aclassify", fake_aclassify)
    monkeypatch.setattr(ingestor, "asplit", fake_asplit)
    monkeypatch.setattr(ingestor, "aembed_text", fake_embed)
    return stack


@pytest.fixture()
def make_document(stack):
    """Factory: a fully 'ingested' synthetic document, no pipeline involved."""

    def _make(
        doc_id: str,
        source: Path | str,
        *,
        namespace: str = "",
        with_vectors: bool = True,
    ):
        source = Path(source)
        artifacts.save_parse(synthetic_parse_result(doc_id, source))
        if not with_vectors:
            return
        chunk = Chunk(
            chunk_id=0, section="body", heading="h", text="hello",
            markdown="hello", embedding_text="[doc › body › h]\n\nhello",
            pages=[1], region_ids={1: [0]},
        )
        artifacts.save_split(doc_id, SplitResult(
            sections=[Section(name="body", pages=[1], chunks=[chunk])],
            pages_used=1,
        ))
        stack.store.upsert_chunks(
            doc_id, [chunk], [vec(1.0)], category="report", namespace=namespace
        )
        artifacts.save_ingest_manifest(doc_id, {
            "store": "SqliteStore",
            "namespace": namespace,
            "dimension": DIM,
            "vector_ids": [f"{doc_id}:0"],
        })

    return _make
