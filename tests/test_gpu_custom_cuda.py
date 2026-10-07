import os

import numpy as np
import pytest

from online_reference import causal_attention_online_fp32
from reference import causal_attention_fp32, max_abs_error, max_rel_error

pytestmark = pytest.mark.gpu

if os.environ.get("REQUIRE_A100") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required for native CUDA tests: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")

from custom_cuda_attention import forward


def require_a100():
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"")
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if os.environ.get("REQUIRE_A100") == "1" and "A100" not in name:
        pytest.fail(f"A100 required, found {name}")
    return name


def make_inputs(B=1, H=1, N=64, D=64, seed=12000):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    return q, k, v


def assert_reference_match(actual, q, k, v):
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    abs_err = max_abs_error(actual, expected)
    rel_err = max_rel_error(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    assert abs_err <= 2e-2 + 2e-2 * float(np.max(np.abs(expected))), (
        f"max_abs={abs_err}, max_rel={rel_err}"
    )


@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("B,H,N", [(1, 1, 64), (1, 4, 128), (2, 2, 64)])
def test_custom_fused_cuda_matches_reference(B, H, N, D):
    require_a100()
    q, k, v = make_inputs(B=B, H=H, N=N, D=D, seed=12000 + B + H + N + D)

    actual = forward(q, k, v)
    assert_reference_match(actual, q, k, v)


def test_custom_fused_cuda_matches_online_recurrence():
    require_a100()
    q, k, v = make_inputs(B=1, H=2, N=128, D=64, seed=12100)

    actual = forward(q, k, v)
    online = causal_attention_online_fp32(q, k, v, tile_q=32, tile_k=32)
    online16 = online.astype(np.float16)

    np.testing.assert_allclose(actual, online16, rtol=2e-2, atol=2e-2)


def test_custom_fused_cuda_single_token_is_exact_v0():
    require_a100()
    q, k, v = make_inputs(B=2, H=4, N=1, D=32, seed=12101)

    actual = forward(q, k, v)
    np.testing.assert_array_equal(actual, v)


def test_custom_fused_cuda_constant_values_are_preserved():
    require_a100()
    q, k, _ = make_inputs(B=1, H=2, N=64, D=128, seed=12102)
    v = np.full_like(q, np.float16(3.25))

    actual = forward(q, k, v)
    np.testing.assert_array_equal(actual, v)


def test_custom_fused_cuda_is_causal():
    require_a100()
    q, k, v = make_inputs(B=1, H=2, N=64, D=64, seed=12103)
    before = forward(q, k, v)

    v2 = v.copy()
    v2[..., 32:, :] = np.float16(123.0)
    after = forward(q, k, v2)

    np.testing.assert_array_equal(before[..., :32, :], after[..., :32, :])


def test_custom_fused_cuda_handles_extreme_finite_logits():
    require_a100()
    D = N = 64
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros_like(q)

    for i in range(N):
        q[..., i, i] = np.float16(64.0)
        k[..., i, i] = np.float16(64.0)

    v = np.zeros_like(q)
    v[..., :, 0] = np.arange(N, dtype=np.float16)

    actual = forward(q, k, v)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[0, 0, :, 0],
        np.arange(N, dtype=np.float16),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_custom_fused_cuda_rejects_nonfinite_inputs(bad):
    require_a100()
    q, k, v = make_inputs()

    q_bad = q.copy()
    q_bad.flat[0] = np.float16(bad)

    with pytest.raises(ValueError, match="NaN|infinity"):
        forward(q_bad, k, v)


def test_custom_fused_cuda_rejects_noncontiguous_input():
    require_a100()
    q, k, v = make_inputs()
    q_bad = np.asfortranarray(q)

    with pytest.raises(ValueError, match="C-contiguous"):
        forward(q_bad, k, v)
