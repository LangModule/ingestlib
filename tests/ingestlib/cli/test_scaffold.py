"""`ingestlib init` — real files in a tmp CWD, round-tripped through the
real config loader. No mocks anywhere."""
import pytest

from ingestlib.cli import main
from ingestlib.config import get_config, reset_config


@pytest.fixture()
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def load_written_config(monkeypatch):
    """Load the config.yaml init just wrote; the session's config singleton is
    dropped afterward and rebuilds lazily once monkeypatch restores env+cwd."""

    def load(path):
        monkeypatch.setenv("INGESTLIB_CONFIG", str(path))
        reset_config()
        return get_config()

    yield load
    reset_config()


def test_init_writes_config_and_env(tmp_cwd):
    assert main(["init"]) == 0
    assert (tmp_cwd / "config.yaml").is_file()
    assert (tmp_cwd / ".env").is_file()
    assert "JINA_API_KEY=" in (tmp_cwd / ".env").read_text()


def test_init_local_writes_only_config(tmp_cwd):
    assert main(["init", "--local"]) == 0
    assert (tmp_cwd / "config.yaml").is_file()
    assert not (tmp_cwd / ".env").exists(), "the zero-cloud preset needs no keys"


def test_init_refuses_to_overwrite(tmp_cwd, capsys):
    (tmp_cwd / "config.yaml").write_text("precious: true\n")
    assert main(["init"]) == 1
    assert (tmp_cwd / "config.yaml").read_text() == "precious: true\n"
    assert "--force" in capsys.readouterr().out


def test_init_force_overwrites(tmp_cwd):
    (tmp_cwd / "config.yaml").write_text("precious: true\n")
    assert main(["init", "--force"]) == 0
    assert "precious" not in (tmp_cwd / "config.yaml").read_text()


def test_local_config_round_trips_through_the_loader(tmp_cwd, load_written_config):
    """The written zero-cloud config must load with NO aws section required."""
    main(["init", "--local"])
    cfg = load_written_config(tmp_cwd / "config.yaml")
    assert cfg.llm_provider == "ollama"
    assert cfg.embedding_provider == "ollama"
    assert cfg.vector_store == "sqlite"
    assert cfg.artifact_store == "local"
    assert cfg.reranker == "none"
    assert cfg.aws.profile == ""  # placeholder identity, nothing uses it


def test_default_config_round_trips_through_the_loader(tmp_cwd, load_written_config):
    main(["init"])
    cfg = load_written_config(tmp_cwd / "config.yaml")
    assert cfg.llm_provider == "bedrock"
    assert cfg.vector_store == "sqlite"
    assert cfg.aws.profile == "your-aws-profile"


def test_next_steps_name_doctor(tmp_cwd, capsys):
    main(["init", "--local"])
    out = capsys.readouterr().out
    assert "ingestlib doctor" in out
    assert "ollama pull" in out


def test_next_steps_include_registry_setup(tmp_cwd, capsys):
    """The corpus path needs the registry — both presets must point the user at it."""
    for args in (["init"], ["init", "--force", "--local"]):
        main(args)
        out = capsys.readouterr().out
        assert "ingestlib registry init" in out
        assert "--profile registry" in out


def test_version_flag_prints_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "ingestlib" in capsys.readouterr().out


def test_console_script_is_registered():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["ingestlib"] == "ingestlib.cli:main"
