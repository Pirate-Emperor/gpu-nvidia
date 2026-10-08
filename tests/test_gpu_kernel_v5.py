"""Correctness tests for experimental shared-tile v5."""

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

from variants.shared_tile_v5 import launch_device


def make_inputs(B, H, N, D, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    return q, k, v


def run(q, k, v, tile_k):
    with cp.cuda.Device(0):
        return cp.asnumpy(
            launch_device(cp.asarray(q), cp.asarray(k), cp.asarray(v), tile_k=tile_k)
        )


@pytest.mark.parametrize("D", (32, 64, 128))
@pytest.mark.parametrize("N", (64, 128, 256))
@pytest.mark.parametrize("tile_k", (16, 32, 64))
def test_v5_matches_independent_reference(N, D, tile_k):
    q, k, v = make_inputs(1, 2, N, D, 50000 + N + D + tile_k)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    actual = run(q, k, v, tile_k)
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)


def test_v5_matches_reference_on_batched_case():
    q, k, v = make_inputs(2, 4, 256, 128, 50100)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    actual = run(q, k, v, 32)
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)


def test_v5_future_kv_invariance():
    q, k, v = make_inputs(1, 2, 128, 64, 50200)
    base = run(q, k, v, 32)
    k2, v2 = k.copy(), v.copy()
    k2[..., 64:, :] = np.float16(91.0)
    v2[..., 64:, :] = np.float16(-83.0)
    changed = run(q, k2, v2, 32)
    np.testing.assert_array_equal(base[..., :64, :], changed[..., :64, :])


def test_v5_long_sequence_smoke():
    q, k, v = make_inputs(1, 2, 1024, 128, 50300)
    actual = run(q, k, v, 64)
    assert actual.shape == q.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
