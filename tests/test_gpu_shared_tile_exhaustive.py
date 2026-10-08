"""Extended A100 correctness matrix for the non-cuTile shared-tile backend."""

from __future__ import annotations

import os

import numpy as np
import pytest

from reference import causal_attention_fp32, passes_mixed_tolerance

pytestmark = pytest.mark.gpu

if os.environ.get("REQUIRE_A100") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required for native CUDA tests: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")

from shared_tile_cuda_attention import forward


def require_a100():
    p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    raw = p.get("name", b"")
    name = raw.decode() if isinstance(raw, bytes) else str(raw)
    if os.environ.get("REQUIRE_A100") == "1" and "A100" not in name:
        pytest.fail(f"A100 required, found {name}")
    return name


def make_inputs(B, H, N, D, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    return q, k, v


def assert_match(actual, q, k, v):
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(
        actual, expected, abs_scale=2e-2, rel_scale=2e-2
    )


@pytest.mark.parametrize("tile_k", (32, 64))
@pytest.mark.parametrize("D", (32, 64, 128))
@pytest.mark.parametrize("N", (64, 128, 256))
def test_shared_tile_matches_reference_across_required_dimensions(tile_k, D, N):
    if N % tile_k:
        pytest.skip("N not divisible by tile")
    require_a100()
    q, k, v = make_inputs(1, 2, N, D, 21000 + D + N + tile_k)
    assert_match(forward(q, k, v, tile_k=tile_k), q, k, v)


@pytest.mark.parametrize("tile_k", (32, 64))
def test_shared_tile_matches_reference_with_batch_head_isolation(tile_k):
    require_a100()
    q, k, v = make_inputs(2, 4, 128, 64, 21100 + tile_k)
    out = forward(q, k, v, tile_k=tile_k)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    np.testing.assert_allclose(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tile_k", (32, 64))
def test_shared_tile_tile_size_invariance(tile_k):
    require_a100()
    q, k, v = make_inputs(1, 2, 256, 128, 21200)
    a = forward(q, k, v, tile_k=32)
    b = forward(q, k, v, tile_k=64)
    np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("D", (32, 64, 128))
def test_shared_tile_zero_logits_are_prefix_means(D):
    require_a100()
    N = 128
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros_like(q)
    v = np.arange(N * D, dtype=np.float32).reshape(1, 1, N, D).astype(np.float16)

    out = forward(q, k, v, tile_k=32)
    expected = np.empty_like(out)
    ref = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)
    for i in (0, 1, 31, 32, 63, 127):
        prefix = v[..., : i + 1, :].astype(np.float32).mean(axis=-2)
        np.testing.assert_allclose(
            out[..., i, :], prefix, rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize("D", (32, 64, 128))
def test_shared_tile_constant_values_are_exact_after_fp32_accumulation(D):
    require_a100()
    q, k, _ = make_inputs(1, 1, 128, D, 21300 + D)
    v = np.full_like(q, np.float16(-2.75))
    out = forward(q, k, v, tile_k=64)
    np.testing.assert_array_equal(out, v)


def test_shared_tile_extreme_logits_are_finite_and_causal():
    require_a100()
    n = d = 128
    q = np.zeros((1, 1, n, d), dtype=np.float16)
    k = np.zeros_like(q)
    for i in range(n):
        q[..., i, i] = np.float16(64.0)
        k[..., i, i] = np.float16(64.0)
    v = np.zeros_like(q)
    v[..., :, 0] = np.arange(n, dtype=np.float16)

    out = forward(q, k, v, tile_k=32)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(
        out[0, 0, :, 0], np.arange(n, dtype=np.float16), rtol=2e-2, atol=2e-2
    )


def test_shared_tile_future_kv_invariance():
    require_a100()
    q, k, v = make_inputs(1, 2, 128, 64, 21400)
    base = forward(q, k, v, tile_k=32)

    k2 = k.copy()
    v2 = v.copy()
    k2[..., 64:, :] = np.float16(91.0)
    v2[..., 64:, :] = np.float16(-83.0)

    changed = forward(q, k2, v2, tile_k=32)
    np.testing.assert_array_equal(base[..., :64, :], changed[..., :64, :])


def test_shared_tile_rejects_nonfinite_without_launch():
    require_a100()
    q, k, v = make_inputs(1, 1, 64, 64, 21500)
    q.flat[0] = np.nan
    with pytest.raises(ValueError, match="NaN|infinity"):
        forward(q, k, v, tile_k=64)

