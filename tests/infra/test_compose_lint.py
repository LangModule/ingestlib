"""Layer C — static lint of infra/docker-compose.yml.

Needs no containers and no docker (pure YAML parse), so it runs in ordinary CI
and guards the compose FILE against regressions: unpinned images, host-port
collisions, orphan volumes, a service missing its healthcheck, profile drift.
"""
import subprocess

import pytest

from tests.infra._util import COMPOSE_PATH, DISTROLESS, docker_available, load_compose, services


def test_every_image_is_pinned():
    for name, svc in services().items():
        image = svc["image"]
        assert ":" in image, f"{name}: image {image!r} has no explicit tag"
        tag = image.rsplit(":", 1)[1]
        assert tag != "latest", f"{name}: image is pinned to :latest — pin a version"


def test_no_host_port_collisions():
    seen: dict[str, str] = {}
    for name, svc in services().items():
        for mapping in svc.get("ports", []):
            parts = str(mapping).split(":")
            host_port = parts[-2] if len(parts) >= 2 else parts[0]  # H:C or IP:H:C
            assert host_port not in seen, (
                f"host port {host_port} published by both {seen[host_port]} and {name}"
            )
            seen[host_port] = name


def test_all_named_volumes_declared():
    compose = load_compose()
    declared = set((compose.get("volumes") or {}).keys())
    for name, svc in compose.get("services", {}).items():
        for mount in svc.get("volumes", []):
            source = str(mount).split(":")[0]
            if source.startswith((".", "/")):
                continue  # bind mount, not a named volume
            assert source in declared, (
                f"{name}: named volume {source!r} is not declared in top-level volumes:"
            )


def test_healthcheck_policy():
    """Every service has a healthcheck except the distroless ones (no shell to run one)."""
    without = {name for name, svc in services().items() if "healthcheck" not in svc}
    assert without == DISTROLESS, (
        f"services without a healthcheck should be exactly {DISTROLESS}, got {without}"
    )


def test_every_service_joins_the_all_profile():
    for name, svc in services().items():
        assert "all" in (svc.get("profiles") or []), f"{name} is not in the 'all' profile"


def test_compose_config_parses():
    if not docker_available():
        pytest.skip("needs the docker CLI to validate interpolation")
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_PATH), "config", "-q"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
