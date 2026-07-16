"""Server-check error hints and pipeline reset — always run, no VL server needed.

_check_server is probed against an unreachable localhost port, so the failure
path is real (connection refused) without depending on any external service.
"""
import pytest

from ingestlib.foundations.ocr import paddle_vl

_DEAD_URL = "http://127.0.0.1:59999/"  # nothing listens here; refusal is immediate


def test_unreachable_server_hints_mlx_launch_command():
    with pytest.raises(RuntimeError, match="mlx_vlm.server") as exc:
        paddle_vl._check_server(_DEAD_URL, backend="mlx-vlm-server")
    assert _DEAD_URL in str(exc.value)
    assert "backend=mlx-vlm-server" in str(exc.value)


def test_unreachable_server_hints_vllm_launch_command():
    with pytest.raises(RuntimeError, match="vllm serve") as exc:
        paddle_vl._check_server(_DEAD_URL, backend="vllm-server")
    assert "mlx_vlm" not in str(exc.value), "NVIDIA users must not be told to start mlx"


def test_reset_pipeline_clears_the_singleton(monkeypatch):
    monkeypatch.setattr(paddle_vl, "_pipeline", object())
    paddle_vl.reset_pipeline()
    assert paddle_vl._pipeline is None
