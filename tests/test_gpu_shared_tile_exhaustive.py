"""Extended A100 correctness matrix for the non-cuTile shared-tile backend."""

from __future__ import annotations

import os

import numpy as np
import pytest

from reference import causal_attention_fp32, passes_mixed_tolerance
from online_reference import causal_attention_online_fp32

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


@pytest.mark.parametrize("tile_k", (16, 32, 64))
@pytest.mark.parametrize("D", (32, 64, 128))
@pytest.mark.parametrize("N", (64, 128, 256))
def test_shared_tile_matches_reference_across_required_dimensions(tile_k, D, N):
    if N % tile_k:
        pytest.skip("N not divisible by tile")
    require_a100()
    q, k, v = make_inputs(1, 2, N, D, 21000 + D + N + tile_k)
    assert_match(forward(q, k, v, tile_k=tile_k), q, k, v)


@pytest.mark.parametrize("tile_k", (16, 32, 64))
def test_shared_tile_matches_reference_with_batch_head_isolation(tile_k):
    require_a100()
    q, k, v = make_inputs(2, 4, 128, 64, 21100 + tile_k)
    out = forward(q, k, v, tile_k=tile_k)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    np.testing.assert_allclose(out, expected, rtol=2e-2, atol=2e-2)


def test_shared_tile_all_tile_sizes_agree():
    require_a100()
    q, k, v = make_inputs(1, 2, 256, 128, 21200)
    outs = [forward(q, k, v, tile_k=tk) for tk in (16, 32, 64)]
    for out in outs[1:]:
        np.testing.assert_allclose(outs[0], out, rtol=2e-2, atol=2e-2)


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



@pytest.mark.parametrize("tile_k", [8, 128, 20])
def test_shared_tile_rejects_unsupported_tile_sizes(tile_k):
    require_a100()
    q, k, v = make_inputs(1, 1, 64, 64, 21700 + tile_k)
    with pytest.raises(ValueError, match="tile"):
        forward(q, k, v, tile_k=tile_k)


def test_shared_tile_long_sequence_matches_online_reference():
    require_a100()
    q, k, v = make_inputs(1, 2, 1024, 128, 21900)
    actual = forward(q, k, v, tile_k=64)
    expected = causal_attention_online_fp32(
        q, k, v, tile_q=4, tile_k=64
    ).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


def test_shared_tile_long_usage_smoke():
    require_a100()
    q, k, v = make_inputs(1, 2, 2048, 128, 21950)
    out = forward(q, k, v, tile_k=64)
    assert out.shape == q.shape
    assert out.dtype == np.float16
    assert np.isfinite(out).all()

    v2 = v.copy()
    v2[..., 1024:, :] = np.float16(77.0)
    out2 = forward(q, k, v2, tile_k=64)
    np.testing.assert_array_equal(out[..., :1024, :], out2[..., :1024, :])


def test_shared_tile_affine_value_transform():
    require_a100()
    q = np.array(
        [[[[1, 0, 1, 0], [0, 1, -1, 0], [1, 1, 0, -1]]]],
        dtype=np.float16,
    )
    k = np.array(
        [[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]],
        dtype=np.float16,
    )
    v = np.array(
        [[[[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]]]],
        dtype=np.float16,
    )
    base = forward(
        np.pad(q, ((0, 0), (0, 0), (0, 1), (0, 0))),
        np.pad(k, ((0, 0), (0, 0), (0, 1), (0, 0))),
        np.pad(v, ((0, 0), (0, 0), (0, 1), (0, 0))),
        tile_k=2,
    )
    q2 = np.pad(q, ((0, 0), (0, 0), (0, 1), (0, 0)))
    k2 = np.pad(k, ((0, 0), (0, 0), (0, 1), (0, 0)))
    v2 = (2 * np.pad(v, ((0, 0), (0, 0), (0, 1), (0, 0))).astype(np.float32) - 3).astype(np.float16)
    transformed = forward(q2, k2, v2, tile_k=2)
    np.testing.assert_allclose(
        transformed,
        (2 * base.astype(np.float32) - 3).astype(np.float16),
        rtol=2e-2,
        atol=2e-2,
    )


def test_shared_tile_current_query_locality():
    require_a100()
    q, k, v = make_inputs(1, 1, 128, 64, 22000)
    before = forward(q, k, v, tile_k=32)
    q2 = q.copy()
    q2[..., 64, :] = np.float16(9.0)
    after = forward(q2, k, v, tile_k=32)
    np.testing.assert_array_equal(before[..., :64, :], after[..., :64, :])
    assert not np.array_equal(before[..., 64, :], after[..., 64, :])


def test_shared_tile_resident_device_matches_host_wrapper():
    require_a100()
    import cupy as cp
    q, k, v = make_inputs(1, 1, 128, 64, 22001)
    host_out = forward(q, k, v, tile_k=64)
    with cp.cuda.Device(0):
        qg = cp.asarray(q)
        kg = cp.asarray(k)
        vg = cp.asarray(v)
        device_out = forward(q, k, v, tile_k=64)
        device_direct = cp.asnumpy(
            __import__("shared_tile_cuda_attention").launch_device(
                qg, kg, vg, tile_k=64
            )
        )
    np.testing.assert_allclose(host_out, device_out, rtol=0, atol=0)
    np.testing.assert_allclose(device_direct, host_out, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("seed", range(5))
def test_shared_tile_randomized_small_differential(seed):
    require_a100()
    n = 64 * (seed % 2 + 1)
    d = (32, 64, 128)[seed % 3]
    q, k, v = make_inputs(1, 1, n, d, 21800 + seed)
    ref = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    for tile_k in (16, 32, 64):
        out = forward(q, k, v, tile_k=tile_k)
        assert passes_mixed_tolerance(out, ref, abs_scale=2e-2, rel_scale=2e-2)
