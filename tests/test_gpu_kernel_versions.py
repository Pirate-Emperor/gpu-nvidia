"""Regression tests for preserved kernel versions v1 and v2."""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

if os.environ.get("REQUIRE_A100") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")

from reference import causal_attention_fp32, passes_mixed_tolerance
from variants.shared_tile_v1 import launch_device as v1_launch
from variants.shared_tile_v2 import launch_device as v2_launch


def make_inputs(N, D, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, 1, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 1, N, D), dtype=np.float32).astype(np.float16)
    return q, k, v


@pytest.mark.parametrize("N,D,tile_k", [
    (64, 32, 16),
    (64, 32, 32),
    (64, 64, 64),
    (128, 64, 32),
    (128, 128, 64),
    (256, 128, 64),
])
def test_preserved_versions_match_reference(N, D, tile_k):
    q, k, v = make_inputs(N, D, 22000 + N + D + tile_k)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    qg, kg, vg = cp.asarray(q), cp.asarray(k), cp.asarray(v)
    v1 = cp.asnumpy(v1_launch(qg, kg, vg, tile_k=tile_k))
    v2 = cp.asnumpy(v2_launch(qg, kg, vg, tile_k=tile_k))

    assert passes_mixed_tolerance(v1, expected, abs_scale=2e-2, rel_scale=2e-2)
    assert passes_mixed_tolerance(v2, expected, abs_scale=2e-2, rel_scale=2e-2)
    np.testing.assert_allclose(v1, v2, rtol=2e-2, atol=2e-2)
