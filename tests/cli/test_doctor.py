"""`ingestlib doctor` checks — the free ones run ungated against reality;
failures are provoked for real (dead ports), never mocked."""
import dataclasses
import sqlite3

import pytest

import ingestlib.config as config_module
from ingestlib.cli import doctor
from ingestlib.config import (
    ArtifactsConfig,
    OllamaConfig,
    PaddleVLConfig,
    SqliteConfig,
    get_config,
    reset_config,
)


@pytest.fixture()
def patched(monkeypatch):
    """Replace config fields for one test; drop every cached client after so
    later tests rebuild against the real config."""

    def apply(**fields):
        cfg = dataclasses.replace(get_config(), **fields)
        monkeypatch.setattr(config_module, "_config", cfg)
        return cfg

    yield apply
    reset_config()


def test_python_check_passes_on_this_interpreter():
    status, detail = doctor.check_python()
    assert status == "ok" and "python 3." in detail


def test_libreoffice_check_never_fails_hard():
    status, detail = doctor.check_libreoffice()
    assert status in ("ok", "warn")
    if status == "warn":
        assert "PDF and images work without" in detail


def test_ocr_server_down_is_a_warning_not_a_failure(patched):
    patched(paddle_vl=PaddleVLConfig(
        backend="mlx-vlm-server",
        server_url="http://localhost:1/",
        api_model_name="PaddlePaddle/PaddleOCR-VL-1.6",
    ))
    status, detail = doctor.check_ocr_server()
    assert status == "warn"
    assert "classify, split, and retrieve" in detail


def test_llm_check_fails_with_the_ollama_hint_on_a_dead_port(patched):
    patched(
        llm_provider="ollama",
        ollama=OllamaConfig(
            base_url="http://localhost:1/v1",
            llm_model_id="qwen3.5:9b",
            embedding_model_id="qwen3-embedding:0.6b",
        ),
    )
    status, detail = doctor.check_llm()
    assert status == "fail"
    assert "ollama" in detail.lower()


def test_reranker_none_is_skipped(patched):
    patched(reranker="none")
    status, detail = doctor.check_reranker()
    assert status == "skip"
    assert "vector order" in detail


def test_local_artifacts_writable(patched, tmp_path):
    patched(artifact_store="local", artifacts=ArtifactsConfig(path=tmp_path / "arts"))
    status, detail = doctor.check_artifact_store()
    assert status == "ok"
    assert not list((tmp_path / "arts").iterdir()), "the write probe must clean up"


def test_sqlite_ping_never_creates_the_schema(patched, tmp_path):
    """A doctor run must not fix the vector dimension before the first ingest."""
    db = tmp_path / "doctor.db"
    patched(vector_store="sqlite", sqlite=SqliteConfig(path=db))
    status, detail = doctor.check_vector_store()
    assert status == "ok"
    tables = sqlite3.connect(db).execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    assert tables == [], "the liveness ping must leave the database schemaless"


def test_unknown_store_fails_cleanly(patched):
    patched(vector_store="chroma")
    status, detail = doctor.check_vector_store()
    assert status == "fail"
    assert "chroma" in detail
