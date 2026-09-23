"""Shared helpers for the infra test suite.

The infra tests verify the CONTAINERS/DBs themselves (reachable, healthy, right
version, required capability, right config) — distinct from the storage tests,
which verify the connector round-trip. Three layers:

  - test_compose_lint.py        static checks on docker-compose.yml (no containers)
  - test_registry / _vector_stores / _sql_sources / _artifact_store   service-contract checks (live)
  - test_runtime_integrity.py   docker-level: health status, image==pin, persistence

Live layers are opt-in via RUN_INFRA_E2E=1 and skip per-service when a service
is unreachable, so a partial `--profile` bring-up still works. Connection details
come from the same env vars the connectors use (defaults match the compose).
"""
import os
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from shutil import which
from urllib.parse import urlsplit, urlunsplit

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = REPO_ROOT / "infra" / "docker-compose.yml"

# images with no shell/curl → no in-container healthcheck is possible
DISTROLESS = {"qdrant", "weaviate"}

# opt-in gate for the live (containers-up) layers
infra_e2e = pytest.mark.skipif(
    os.environ.get("RUN_INFRA_E2E") != "1",
    reason="infra e2e is opt-in: set RUN_INFRA_E2E=1 with the containers up",
)


def load_compose() -> dict:
    with open(COMPOSE_PATH) as f:
        return yaml.safe_load(f)


def services() -> dict:
    return load_compose().get("services", {})


def http_get(url: str, timeout: float = 5.0):
    """(status_code, body_bytes), or (None, None) when unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception:
        return None, None


def reachable_http(url: str, timeout: float = 3.0) -> bool:
    return http_get(url, timeout)[0] is not None


def tcp_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def pg_url_plain(url: str) -> str:
    """The DSN in the plain postgresql:// form psycopg.connect accepts."""
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def ro_url_from(rw_url: str) -> str:
    """Derive the read-only DSN from the read-write one (compose creds), unless
    INGESTLIB_REGISTRY_RO_URL overrides it for a bring-your-own Postgres."""
    override = os.environ.get("INGESTLIB_REGISTRY_RO_URL")
    if override:
        return pg_url_plain(override)
    parts = urlsplit(pg_url_plain(rw_url))
    netloc = f"ingestlib_ro:ro_pw@{parts.hostname}:{parts.port or 5432}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def docker_available() -> bool:
    return which("docker") is not None


def docker_inspect(container: str, fmt: str) -> str | None:
    """`docker inspect --format` output, or None when the container is absent."""
    out = subprocess.run(
        ["docker", "inspect", "--format", fmt, container],
        capture_output=True, text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else None
