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

from custom_cuda_attention import forward as rowwise_forward
from shared_tile_cuda_attention import forward as shared_tile_forward
from warp_cuda_attention import forward as warp_forward


BACKENDS = {
    "rowwise_block": rowwise_forward,
    "warp_row": warp_forward,
}


def require_a100():
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"")
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if os.environ.get("REQUIRE_A100") == "1" and "A100" not in name:
        pytest.fail(f"A100 required, found {name}")
    return name


def make_inputs(B=1, H=1, N=64, D=64, seed=13000):
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


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
@pytest.mark.parametrize(
    "B,H,N,D",
    [
        (1, 1, 64, 32),
        (1, 1, 64, 64),
        (1, 1, 64, 128),
        (1, 4, 128, 64),
        (2, 2, 64, 128),
    ],
)
def test_all_custom_cuda_variants_match_independent_reference(backend, B, H, N, D):
    require_a100()
    q, k, v = make_inputs(B=B, H=H, N=N, D=D, seed=13000 + B + H + N + D)
    actual = BACKENDS[backend](q, k, v)
    assert_reference_match(actual, q, k, v)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_custom_variants_match_online_recurrence(backend):
    require_a100()
    q, k, v = make_inputs(B=1, H=2, N=128, D=64, seed=13100)

    actual = BACKENDS[backend](q, k, v)
    expected = causal_attention_online_fp32(q, k, v, tile_q=32, tile_k=32)
    np.testing.assert_allclose(
        actual, expected.astype(np.float16), rtol=2e-2, atol=2e-2
    )


def test_independent_custom_cuda_variants_agree():
    require_a100()
    q, k, v = make_inputs(B=2, H=4, N=128, D=128, seed=13101)

    rowwise = rowwise_forward(q, k, v)
    warp = warp_forward(q, k, v)

    np.testing.assert_allclose(rowwise, warp, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_single_token_is_exact_v(backend):
    require_a100()
    q, k, v = make_inputs(B=2, H=4, N=1, D=32, seed=13102)

    actual = BACKENDS[backend](q, k, v)
    np.testing.assert_array_equal(actual, v)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_constant_v_is_preserved(backend):
    require_a100()
    q, k, _ = make_inputs(B=1, H=2, N=64, D=128, seed=13103)
    v = np.full_like(q, np.float16(-2.75))

    actual = BACKENDS[backend](q, k, v)
    np.testing.assert_array_equal(actual, v)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_causality_holds(backend):
    require_a100()
    q, k, v = make_inputs(B=1, H=2, N=64, D=64, seed=13104)

    before = BACKENDS[backend](q, k, v)
    v2 = v.copy()
    v2[..., 32:, :] = np.float16(123.0)
    after = BACKENDS[backend](q, k, v2)

    np.testing.assert_array_equal(before[..., :32, :], after[..., :32, :])


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_extreme_logits_remain_finite(backend):
    require_a100()
    D = N = 64
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros_like(q)

    for i in range(N):
        q[..., i, i] = np.float16(64.0)
        k[..., i, i] = np.float16(64.0)

    v = np.zeros_like(q)
    v[..., :, 0] = np.arange(N, dtype=np.float16)

    actual = BACKENDS[backend](q, k, v)

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[0, 0, :, 0], np.arange(N, dtype=np.float16), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(backend, bad):
    require_a100()
    q, k, v = make_inputs()

    q_bad = q.copy()
    q_bad.flat[0] = np.float16(bad)

    with pytest.raises(ValueError, match="NaN|infinity"):
        BACKENDS[backend](q_bad, k, v)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_noncontiguous_inputs_rejected(backend):
    require_a100()
    q, k, v = make_inputs()
    q_bad = np.asfortranarray(q)

    with pytest.raises(ValueError, match="C-contiguous"):
        BACKENDS[backend](q_bad, k, v)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_identical_keys_reduce_to_prefix_means(backend):
    require_a100()
    q, _, v = make_inputs(B=1, H=2, N=64, D=64, seed=13105)
    k = np.empty_like(q)
    k[...] = np.float16(0.75)

    actual = BACKENDS[backend](q, k, v)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS.keys())
def test_common_key_shift_is_softmax_invariant(backend):
    require_a100()
    q, k, v = make_inputs(B=1, H=2, N=64, D=64, seed=13106)
    shift = np.linspace(-2.0, 2.0, 64, dtype=np.float32).astype(np.float16)

    baseline = BACKENDS[backend](q, k, v)
    shifted = BACKENDS[backend](q, k + shift, v)

    np.testing.assert_allclose(shifted, baseline, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tile_k", [32, 64])
@pytest.mark.parametrize(
    "B,H,N,D",
    [
        (1, 1, 64, 32),
        (1, 1, 64, 64),
        (1, 1, 128, 128),
        (2, 2, 128, 64),
    ],
)
def test_shared_tile_cuda_matches_reference(B, H, N, D, tile_k):
    require_a100()
    q, k, v = make_inputs(B=B, H=H, N=N, D=D, seed=13200 + D + tile_k)

    actual = shared_tile_forward(q, k, v, tile_k=tile_k)
    assert_reference_match(actual, q, k, v)


@pytest.mark.parametrize("tile_k", [32, 64])
def test_shared_tile_cuda_matches_online_recurrence(tile_k):
    require_a100()
    q, k, v = make_inputs(B=1, H=2, N=128, D=128, seed=13300 + tile_k)

    actual = shared_tile_forward(q, k, v, tile_k=tile_k)
    expected = causal_attention_online_fp32(
        q, k, v, tile_q=32, tile_k=tile_k
    )

    np.testing.assert_allclose(
        actual, expected.astype(np.float16), rtol=2e-2, atol=2e-2
    )


def test_all_three_custom_fused_variants_agree():
    require_a100()
    q, k, v = make_inputs(B=1, H=4, N=128, D=64, seed=13400)

    block = rowwise_forward(q, k, v)
    warp = warp_forward(q, k, v)
    tiled = shared_tile_forward(q, k, v, tile_k=64)

    np.testing.assert_allclose(block, warp, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(block, tiled, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tile_k", [32, 64])
def test_shared_tile_rejects_nondivisible_sequences(tile_k):
    require_a100()
    q, k, v = make_inputs(B=1, H=1, N=100, D=64, seed=13500 + tile_k)

    with pytest.raises(ValueError, match="divisible"):
        shared_tile_forward(q, k, v, tile_k=tile_k)
