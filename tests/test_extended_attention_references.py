import numpy as np
import pytest

from extended_attention import (
    causal_attention_gqa_fp32,
    causal_attention_packed_fp32,
)
from reference import AttentionValidationError, causal_attention_fp32


def random_inputs(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32).astype(np.float16)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_gqa_mqa_reference_matches_independent_per_head_reference(kv_heads):
    batch, query_heads, n, d = 1, 4, 64, 32
    q = random_inputs((batch, query_heads, n, d), 100 + kv_heads)
    k = random_inputs((batch, kv_heads, n, d), 200 + kv_heads)
    v = random_inputs((batch, kv_heads, n, d), 300 + kv_heads)

    actual = causal_attention_gqa_fp32(q, k, v, output_dtype=np.float32)
    expected = np.empty((batch, query_heads, n, d), dtype=np.float32)
    group = query_heads // kv_heads
    for hq in range(query_heads):
        hkv = hq // group
        expected[:, hq:hq + 1] = causal_attention_fp32(
            np.ascontiguousarray(q[:, hq:hq + 1]),
            np.ascontiguousarray(k[:, hkv:hkv + 1]),
            np.ascontiguousarray(v[:, hkv:hkv + 1]),
            output_dtype=np.float32,
        )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_packed_varlen_reference_resets_causality_at_every_segment():
    batch, heads, total, d = 1, 2, 192, 32
    q = random_inputs((batch, heads, total, d), 410)
    k = random_inputs((batch, heads, total, d), 411)
    v = random_inputs((batch, heads, total, d), 412)
    offsets = np.array([0, 64, 192], dtype=np.int32)

    actual = causal_attention_packed_fp32(q, k, v, offsets)
    first = causal_attention_fp32(q[..., :64, :].copy(), k[..., :64, :].copy(), v[..., :64, :].copy())
    second = causal_attention_fp32(q[..., 64:, :].copy(), k[..., 64:, :].copy(), v[..., 64:, :].copy())
    expected = np.concatenate((first, second), axis=-2)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    # Changing all values in segment two cannot influence segment one's outputs.
    changed_v = v.copy()
    changed_v[..., 64:, :] = np.float16(123.0)
    changed = causal_attention_packed_fp32(q, k, changed_v, offsets)
    np.testing.assert_array_equal(actual[..., :64, :], changed[..., :64, :])


@pytest.mark.parametrize("offsets", [[1, 64, 192], [0, 64, 191], [0, 64, 64, 192], [0, 63, 192]])
def test_packed_varlen_reference_rejects_invalid_offsets(offsets):
    q = random_inputs((1, 1, 192, 32), 700)
    with pytest.raises(AttentionValidationError):
        causal_attention_packed_fp32(q, q.copy(), q.copy(), np.asarray(offsets))


def test_gqa_reference_rejects_nondivisible_head_counts():
    q = random_inputs((1, 3, 64, 32), 801)
    k = random_inputs((1, 2, 64, 32), 802)
    v = random_inputs((1, 2, 64, 32), 803)
    with pytest.raises(AttentionValidationError, match="divisible"):
        causal_attention_gqa_fp32(q, k, v)
