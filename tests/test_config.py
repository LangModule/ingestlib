"""Config loading, defaults, and reset_config() — pure, always run.

Uses a scratch config dir activated via INGESTLIB_CONFIG. The root conftest
already materialized the real config, and reset_config() mutates process-wide
state (the cache, the dotenv bookkeeping, os.environ) — the fixture snapshots
all three and restores them so the rest of the suite keeps its real config.
"""
import os
from pathlib import Path

import pytest

import ingestlib.config as config_module
from ingestlib.config import get_config, reset_config

_AWS_ONLY = """\
aws:
  profile: test-profile
  region: us-east-1
  account_id: "123456789012"
"""


@pytest.fixture()
def scratch_config(tmp_path):
    """A scratch config dir, active via INGESTLIB_CONFIG, starting from a shell
    that exported none of the real .env's keys. Restores everything after."""
    env_before = dict(os.environ)
    dotenv_keys_before = set(config_module._dotenv_keys)
    config_before = config_module._config

    for key in dotenv_keys_before:  # forget what the real .env injected
        os.environ.pop(key, None)
    config_module._dotenv_keys.clear()
    config_module._config = None
    os.environ["INGESTLIB_CONFIG"] = str(tmp_path / "config.yaml")
    try:
        yield tmp_path
    finally:
        config_module._config = config_before
        config_module._dotenv_keys.clear()
        config_module._dotenv_keys.update(dotenv_keys_before)
        os.environ.clear()
        os.environ.update(env_before)


def _write(directory: Path, yaml_text: str, env_text: str | None = None) -> None:
    (directory / "config.yaml").write_text(yaml_text)
    if env_text is not None:
        (directory / ".env").write_text(env_text)


def test_aws_only_config_loads_with_all_defaults(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    cfg = get_config()
    assert cfg.aws.profile == "test-profile"
    assert cfg.aws.account_id == "123456789012"
    assert cfg.bedrock.llm_model_id == "us.amazon.nova-2-lite-v1:0"
    assert cfg.bedrock.embedding_model_id == "amazon.nova-2-multimodal-embeddings-v1:0"
    assert cfg.bedrock.rerank_model_id == "amazon.rerank-v1:0"
    assert cfg.bedrock.rerank_region == "us-west-2"
    assert cfg.jina.rerank_model_id == "jina-reranker-v3"
    assert cfg.openai.llm_model_id == "gpt-5-mini"
    assert cfg.openai.embedding_model_id == "text-embedding-3-small"
    assert cfg.paddle_vl.api_model_name == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert cfg.s3.bucket == "ingestlib-123456789012"
    assert cfg.vector_store == "pinecone"
    assert cfg.reranker == "jina"
    assert cfg.artifact_store == "s3"
    assert cfg.llm_provider == "bedrock"
    assert cfg.embedding_provider == "bedrock"


def test_artifacts_path_anchors_beside_config(scratch_config):
    _write(scratch_config, _AWS_ONLY + "artifact_store: local\n")
    cfg = get_config()
    assert cfg.artifact_store == "local"
    assert cfg.artifacts.path == (scratch_config / "artifacts").resolve()


def test_missing_aws_section_raises(scratch_config):
    _write(scratch_config, "vector_store: sqlite\n")
    with pytest.raises(KeyError):
        get_config()


def test_reranker_and_vector_store_keys_are_read(scratch_config):
    _write(scratch_config, _AWS_ONLY + "vector_store: sqlite\nreranker: aws\n")
    cfg = get_config()
    assert cfg.vector_store == "sqlite"
    assert cfg.reranker == "aws"


def test_classify_preset_defaults_to_open_ended_without_rules_yaml(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    cfg = get_config()
    assert cfg.classify.rules == {}
    assert cfg.classify.target_pages == ""
    assert cfg.classify.max_pages == 0


def test_classify_preset_is_read_from_rules_yaml(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    (scratch_config / "rules.yaml").write_text(
        "classify:\n"
        "  max_pages: 5\n"
        '  target_pages: "1,3,5-7"\n'
        "  rules:\n"
        "    invoice: Itemized charges and payment terms\n"
        "    sec_filing:\n"           # a rule with no description stays usable
    )
    cfg = get_config()
    assert cfg.classify.max_pages == 5
    assert cfg.classify.target_pages == "1,3,5-7"
    assert cfg.classify.rules == {
        "invoice": "Itemized charges and payment terms",
        "sec_filing": "",
    }


def test_classify_section_in_config_yaml_is_not_read(scratch_config):
    """Rules have exactly one home — a classify block in config.yaml is inert."""
    _write(scratch_config, _AWS_ONLY + "classify:\n  rules:\n    invoice: x\n")
    assert get_config().classify.rules == {}


def test_split_preset_defaults_without_rules_yaml(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    cfg = get_config()
    assert cfg.split.categories == {}
    assert cfg.split.unmatched == "other"


def test_split_preset_is_read_from_rules_yaml(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    (scratch_config / "rules.yaml").write_text(
        "split:\n"
        "  unmatched: skip\n"
        "  categories:\n"
        "    financial_statements: Balance sheets and income statements\n"
        "    notes:\n"                # a category with no description stays usable
    )
    cfg = get_config()
    assert cfg.split.unmatched == "skip"
    assert cfg.split.categories == {
        "financial_statements": "Balance sheets and income statements",
        "notes": "",
    }


def test_reset_config_picks_up_rules_yaml_edits(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    assert get_config().classify.rules == {}
    (scratch_config / "rules.yaml").write_text("classify:\n  rules:\n    invoice: x\n")
    reset_config()
    assert get_config().classify.rules == {"invoice": "x"}


def test_provider_keys_are_read(scratch_config):
    _write(
        scratch_config,
        _AWS_ONLY + "llm_provider: openai\nembedding_provider: openai\n",
    )
    cfg = get_config()
    assert cfg.llm_provider == "openai"
    assert cfg.embedding_provider == "openai"


def test_secrets_load_from_dotenv_next_to_config(scratch_config):
    _write(scratch_config, _AWS_ONLY, "JINA_API_KEY=from-dotenv\nOPENAI_API_KEY=oa-dotenv\n")
    cfg = get_config()
    assert cfg.jina.api_key == "from-dotenv"
    assert cfg.openai.api_key == "oa-dotenv"


def test_config_is_cached_until_reset(scratch_config):
    _write(scratch_config, _AWS_ONLY)
    first = get_config()
    _write(scratch_config, _AWS_ONLY + "reranker: aws\n")
    assert get_config() is first, "edits must not apply without reset_config()"
    reset_config()
    assert get_config().reranker == "aws"


def test_reset_config_applies_dotenv_edits_and_deletions(scratch_config):
    _write(scratch_config, _AWS_ONLY, "JINA_API_KEY=v1\nPINECONE_API_KEY=p1\n")
    cfg = get_config()
    assert cfg.jina.api_key == "v1" and cfg.pinecone.api_key == "p1"

    _write(scratch_config, _AWS_ONLY, "JINA_API_KEY=v2\n")  # edited + deleted
    reset_config()
    cfg = get_config()
    assert cfg.jina.api_key == "v2", "an edited secret must apply on reload"
    assert cfg.pinecone.api_key == "", "a deleted secret must disappear on reload"


def test_shell_exported_vars_survive_reset_and_win_over_dotenv(scratch_config):
    os.environ["JINA_API_KEY"] = "from-shell"
    _write(scratch_config, _AWS_ONLY, "JINA_API_KEY=from-dotenv\n")
    assert get_config().jina.api_key == "from-shell"
    reset_config()
    assert os.environ["JINA_API_KEY"] == "from-shell"
    assert get_config().jina.api_key == "from-shell"


def test_reset_config_clears_imported_client_singletons(scratch_config, monkeypatch):
    from ingestlib.storage.s3 import client as s3_client

    _write(scratch_config, _AWS_ONLY)
    get_config()
    monkeypatch.setattr(s3_client, "_s3_client", object())
    reset_config()
    assert s3_client._s3_client is None, "reset must clear live client singletons"


def test_reset_config_clears_sqlite_schema_state(scratch_config, monkeypatch):
    from ingestlib.storage.sqlite import client as sqlite_client

    _write(scratch_config, _AWS_ONLY)
    get_config()
    monkeypatch.setitem(sqlite_client._ready, Path("/tmp/x.db"), 1024)
    reset_config()
    assert sqlite_client._ready == {}, "reset must drop verified-schema state"


def test_reset_config_clears_openai_model_caches(scratch_config, monkeypatch):
    from ingestlib.foundations.llm.openai import embedding as oa_embedding
    from ingestlib.foundations.llm.openai import mini as oa_mini

    _write(scratch_config, _AWS_ONLY)
    get_config()
    monkeypatch.setitem(oa_mini._model_cache, "stale", object())
    monkeypatch.setitem(oa_embedding._embedder_cache, 1024, object())
    reset_config()
    assert oa_mini._model_cache == {}, "reset must drop cached chat models"
    assert oa_embedding._embedder_cache == {}, "reset must drop cached embedders"
