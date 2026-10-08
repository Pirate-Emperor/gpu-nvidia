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

from custom_cuda_attention import forward as block_forward
from shared_tile_cuda_attention import forward as tiled_forward


def gpu_name(device_id):
    raw = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


@pytest.mark.parametrize("device_id", [0, 1])
def test_selected_a100_devices_execute_native_cuda(device_id):
    count = cp.cuda.runtime.getDeviceCount()
    if device_id >= count:
        pytest.skip(f"device {device_id} is not exposed to this runner")
    with cp.cuda.Device(device_id):
        name = gpu_name(device_id)
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
        out = block_forward(q, k, v)

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
            outs.append(block_forward(q, k, v))

    np.testing.assert_allclose(outs[0], outs[1], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device_id", [0, 1])
@pytest.mark.parametrize("tile_k", [32, 64])
def test_selected_a100_devices_run_shared_tile_v2(device_id, tile_k):
    count = cp.cuda.runtime.getDeviceCount()
    if device_id >= count:
        pytest.skip(f"device {device_id} is not exposed to this runner")

    rng = np.random.default_rng(17100 + device_id + tile_k)
    q = rng.standard_normal((1, 2, 128, 64), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 2, 128, 64), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 2, 128, 64), dtype=np.float32).astype(np.float16)

    with cp.cuda.Device(device_id):
        out = tiled_forward(q, k, v, tile_k=tile_k)

    assert out.shape == q.shape
    assert out.dtype == np.float16
    assert np.isfinite(out).all()


def test_both_exposed_devices_produce_consistent_shared_tile_results():
    count = cp.cuda.runtime.getDeviceCount()
    if count < 2:
        pytest.skip("runner exposes fewer than two CUDA devices")

    rng = np.random.default_rng(17003)
    q = rng.standard_normal((1, 2, 128, 64), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 2, 128, 64), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 2, 128, 64), dtype=np.float32).astype(np.float16)

    outs = []
    for device_id in (0, 1):
        with cp.cuda.Device(device_id):
            outs.append(tiled_forward(q, k, v, tile_k=64))

    np.testing.assert_allclose(outs[0], outs[1], rtol=2e-2, atol=2e-2)
