import numpy as np

from reference import causal_attention_fp32


def make_inputs(B=1, H=2, N=32, D=64, seed=14000):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    return q, k, v


def test_identical_keys_reduce_attention_to_prefix_value_means():
    q, _, v = make_inputs(B=2, H=4, N=32, D=64, seed=14001)
    k = np.empty_like(q)
    k[..., 0, :] = np.float16(0.75)
    for j in range(1, q.shape[-2]):
        k[..., j, :] = k[..., 0, :]

    out = causal_attention_fp32(q, k, v)

    for row in [0, 1, 7, 15, 31]:
        expected = np.mean(v[..., : row + 1, :].astype(np.float32), axis=-2)
        np.testing.assert_allclose(out[..., row, :], expected, rtol=1e-6, atol=1e-6)


def test_adding_common_key_vector_does_not_change_attention():
    q, k, v = make_inputs(seed=14002)
    shift = np.linspace(-2.0, 2.0, q.shape[-1], dtype=np.float32).astype(np.float16)

    baseline = causal_attention_fp32(q, k, v)
    shifted = causal_attention_fp32(q, k + shift, v)

    # Q/K are stored in FP16, so forming K+shift introduces FP16 rounding.
    # The invariant is exact in real arithmetic; compare at the project's
    # FP16-input numerical scale rather than requiring bitwise equality.
    np.testing.assert_allclose(shifted, baseline, rtol=1e-3, atol=2e-3)


def test_attention_is_affine_in_values():
    q, k, v = make_inputs(seed=14003)
    alpha = np.float32(1.75)
    beta = np.float32(-0.5)

    base = causal_attention_fp32(q, k, v)
    transformed_v = (alpha * v.astype(np.float32) + beta).astype(np.float16)
    transformed = causal_attention_fp32(q, k, transformed_v)

    predicted = alpha * base + beta
    np.testing.assert_allclose(transformed, predicted, rtol=2e-3, atol=2e-3)
