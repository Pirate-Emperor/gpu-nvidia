import numpy as np
import pytest

from reference import causal_attention_fp32, max_abs_error, max_rel_error

@pytest.mark.parametrize(
    "B,H,N,D,seed",
    [
        (1, 1, 1, 32, 100),
        (1, 2, 32, 32, 101),
        (2, 1, 64, 64, 102),
        (2, 4, 64, 128, 103),
        (1, 8, 128, 64, 104),
        (2, 8, 128, 128, 105),
    ],
)
def test_reference_randomized_required_shapes_are_finite(B, H, N, D, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N), dtype=np.float32).astype(np.float16).reshape(B,H,N,1)
    v = np.repeat(v, D, axis=-1)

    out = causal_attention_fp32(q, k, v)

    assert out.shape == (B, H, N, D)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()

@pytest.mark.parametrize("D", [32, 64, 128])
def test_zero_logits_reduce_to_prefix_means(D):
    N = 64
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros_like(q)

    token_values = np.arange(N * D, dtype=np.float32).reshape(1, 1, N, D)
    v = token_values.astype(np.float16)

    out = causal_attention_fp32(q, k, v)

    for row in [0, 1, 7, 31, 63]:
        expected = np.mean(v[..., : row + 1, :].astype(np.float32), axis=-2)
        np.testing.assert_allclose(out[..., row, :], expected, rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("D", [32, 64, 128])
def test_constant_value_tensor_is_preserved(D):
    N = 64
    rng = np.random.default_rng(200 + D)
    q = rng.standard_normal((1, 1, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, N, D), dtype=np.float32).astype(np.float16)
    v = np.full((1, 1, N, D), np.float16(3.25), dtype=np.float16)

    out = causal_attention_fp32(q, k, v)

    np.testing.assert_array_equal(out, np.full_like(out, 3.25, dtype=np.float32))

def test_extreme_finite_logits_remain_stable():
    D = 64
    N = 64
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros_like(q)
    q[..., :, 0] = np.float16(64.0)
    k[..., :, 0] = np.float16(64.0)
    v = np.zeros((1, 1, N, D), dtype=np.float16)
    v[..., :, 0] = np.arange(N, dtype=np.float16)

    out = causal_attention_fp32(q, k, v)

    assert np.isfinite(out).all()
    assert np.isfinite(max_abs_error(out, out))
    assert np.isfinite(max_rel_error(out, out))
    np.testing.assert_allclose(out[..., :, 0], np.arange(N, dtype=np.float32), atol=1e-5)

def test_output_stays_inside_prefix_value_convex_hull():
    rng = np.random.default_rng(300)
    q = rng.standard_normal((1, 2, 32, 32), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 2, 32, 32), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 2, 32, 32), dtype=np.float32).astype(np.float16)

    out = causal_attention_fp32(q, k, v)

    v32 = v.astype(np.float32)
    for row in [0, 1, 7, 15, 31]:
        prefix = v32[..., : row + 1, :]
        lower = prefix.min(axis=-2)
        upper = prefix.max(axis=-2)
        assert np.all(out[..., row, :] >= lower - 1e-5)
        assert np.all(out[..., row, :] <= upper + 1e-5)

def test_large_head_count_and_batch_count_match_repeated_single_head_semantics():
    rng = np.random.default_rng(301)
    q = rng.standard_normal((2, 4, 32, 32), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((2, 4, 32, 32), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((2, 4, 32, 32), dtype=np.float32).astype(np.float16)

    batched = causal_attention_fp32(q, k, v)

    for b in range(2):
        for h in range(4):
            single = causal_attention_fp32(
                q[b : b + 1, h : h + 1],
                k[b : b + 1, h : h + 1],
                v[b : b + 1, h : h + 1],
            )
            np.testing.assert_array_equal(batched[b : b + 1, h : h + 1], single)

def test_output_changes_when_only_current_query_changes():
    rng = np.random.default_rng(302)
    q = rng.standard_normal((1, 1, 16, 32), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, 16, 32), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 1, 16, 32), dtype=np.float32).astype(np.float16)

    before = causal_attention_fp32(q, k, v)
    q2 = q.copy()
    q2[..., 8, :] = np.float16(12.0)
    after = causal_attention_fp32(q2, k, v)

    np.testing.assert_array_equal(before[..., :8, :], after[..., :8, :])
    assert not np.array_equal(before[..., 8, :], after[..., 8, :])