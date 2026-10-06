import os

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


if os.environ.get("REQUIRE_A100") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required for native CUDA tests: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")

from custom_cuda_attention import forward


def gpu_name():
    raw = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


@pytest.mark.parametrize("device_id", [0, 1])
def test_selected_a100_devices_execute_native_cuda(device_id):
    count = cp.cuda.runtime.getDeviceCount()
    if device_id >= count:
        pytest.skip(f"device {device_id} is not exposed to this runner")
    with cp.cuda.Device(device_id):
        name = gpu_name()
        if os.environ.get("REQUIRE_A100") == "1":
            assert "A100" in name
        x = cp.arange(1024, dtype=cp.float32)
        y = (x * 3.0 + 2.0).sum()
        assert float(y.get()) == float(3 * 1024 * 1023 / 2 + 2 * 1024)


@pytest.mark.parametrize("device_id", [0, 1])
def test_selected_a100_devices_run_fused_attention(device_id):
    count = cp.cuda.runtime.getDeviceCount()
    if device_id >= count:
        pytest.skip(f"device {device_id} is not exposed to this runner")

    rng = np.random.default_rng(17000 + device_id)
    q = rng.standard_normal((1, 1, 64, 32), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, 64, 32), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 1, 64, 32), dtype=np.float32).astype(np.float16)

    with cp.cuda.Device(device_id):
        out = forward(q, k, v)

    assert out.shape == q.shape
    assert out.dtype == np.float16
    assert np.isfinite(out).all()


def test_both_exposed_devices_produce_consistent_results():
    count = cp.cuda.runtime.getDeviceCount()
    if count < 2:
        pytest.skip("runner exposes fewer than two CUDA devices")

    rng = np.random.default_rng(17002)
    q = rng.standard_normal((1, 2, 64, 64), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 2, 64, 64), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 2, 64, 64), dtype=np.float32).astype(np.float16)

    outs = []
    for device_id in (0, 1):
        with cp.cuda.Device(device_id):
            outs.append(forward(q, k, v))

    np.testing.assert_allclose(outs[0], outs[1], rtol=2e-2, atol=2e-2)
