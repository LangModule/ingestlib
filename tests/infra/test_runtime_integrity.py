"""Layer B — runtime integrity of the running compose (needs the docker CLI).

Guards what the static lint can't: that each container is actually `healthy`
(covers etcd/minio, which have no host port), that the RUNNING image equals the
pin in the compose (catches drift), that init.sql is safe to re-run, and — the
property the volume gotchas exist for — that data survives a container restart.

Opt-in via RUN_INFRA_E2E=1 + the docker CLI. The restart test is doubly opt-in
(RUN_INFRA_RESTART=1) because it restarts the registry container.
"""
import os
import subprocess
import time

import pytest

from tests.infra._util import DISTROLESS, docker_available, docker_inspect, infra_e2e, pg_url_plain, services

pytestmark = [
    infra_e2e,
    pytest.mark.skipif(not docker_available(), reason="needs the docker CLI"),
]

_SERVICES = list(services().items())


@pytest.mark.parametrize("name,svc", _SERVICES, ids=[n for n, _ in _SERVICES])
def test_container_running_and_healthy(name, svc):
    container = svc["container_name"]
    state = docker_inspect(container, "{{.State.Status}}")
    if state is None:
        pytest.skip(f"{container} not created (its profile isn't up)")
    assert state == "running", f"{container} is {state}, not running"
    if name in DISTROLESS:
        return  # distroless: no healthcheck by design
    health = docker_inspect(container, "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}")
    assert health == "healthy", f"{container} health is {health!r}"


@pytest.mark.parametrize("name,svc", _SERVICES, ids=[n for n, _ in _SERVICES])
def test_running_image_matches_pin(name, svc):
    container = svc["container_name"]
    running = docker_inspect(container, "{{.Config.Image}}")
    if running is None:
        pytest.skip(f"{container} not created (its profile isn't up)")
    assert running == svc["image"], f"{container} runs {running!r}, compose pins {svc['image']!r}"


def test_init_sql_is_idempotent():
    """Re-running the mounted init.sql must not error — the CREATE ROLE guard we added."""
    container = services()["registry"]["container_name"]
    if docker_inspect(container, "{{.State.Status}}") != "running":
        pytest.skip("registry container not up")
    cmd = [
        "docker", "exec", container, "psql", "-U", "ingestlib", "-d", "ingestlib",
        "-v", "ON_ERROR_STOP=1", "-f", "/docker-entrypoint-initdb.d/init.sql",
    ]
    for attempt in (1, 2):
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"init.sql run #{attempt} failed: {result.stderr[-300:]}"


@pytest.mark.skipif(
    os.environ.get("RUN_INFRA_RESTART") != "1",
    reason="restart test is opt-in: set RUN_INFRA_RESTART=1 (it restarts the registry container)",
)
def test_registry_data_survives_restart():
    """A named volume must persist data across a container recreate — the reason
    the pg18/mongo/weaviate volume-path fixes exist. Written, restarted, re-read."""
    psycopg = pytest.importorskip("psycopg")
    container = services()["registry"]["container_name"]
    if docker_inspect(container, "{{.State.Status}}") != "running":
        pytest.skip("registry container not up")
    url = pg_url_plain(
        os.environ.get("INGESTLIB_REGISTRY_URL", "postgresql://ingestlib:pw@localhost:5433/ingestlib")
    )
    marker = "infra-persist-probe"

    with psycopg.connect(url, autocommit=True) as c:
        c.execute("INSERT INTO documents (doc_id) VALUES (%s) ON CONFLICT DO NOTHING", (marker,))
    try:
        subprocess.run(["docker", "restart", container], check=True, capture_output=True)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if docker_inspect(container, "{{.State.Health.Status}}") == "healthy":
                break
            time.sleep(2)
        else:
            pytest.fail("registry did not return to healthy within 90s after restart")
        with psycopg.connect(url, autocommit=True) as c:
            assert c.execute(
                "SELECT count(*) FROM documents WHERE doc_id=%s", (marker,)
            ).fetchone()[0] == 1, "marker row did not survive the restart"
    finally:
        with psycopg.connect(url, autocommit=True) as c:
            c.execute("DELETE FROM documents WHERE doc_id=%s", (marker,))
