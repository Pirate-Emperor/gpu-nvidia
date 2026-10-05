import numpy as np
import pytest

from reference import (
    AttentionValidationError,
    causal_attention_fp32,
    max_abs_error,
    max_rel_error,
    passes_mixed_tolerance,
    validate_attention_inputs,
)


def make_inputs(B=1, H=2, N=4, D=8, Dv=None, seed=1234):
    Dv = D if Dv is None else Dv
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, Dv), dtype=np.float32).astype(np.float16)
    return q, k, v


def test_reference_shape_and_finiteness():
    q, k, v = make_inputs(N=8, D=64)
    out = causal_attention_fp32(q, k, v)
    assert out.shape == (1, 2, 8, 64)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"B": 1, "H": 1, "N": 1, "D": 64},
        {"B": 2, "H": 1, "N": 64, "D": 64},
        {"B": 1, "H": 4, "N": 128, "D": 128},
    ],
)
def test_reference_required_head_dimensions(kwargs):
    q, k, v = make_inputs(**kwargs)
    out = causal_attention_fp32(q, k, v)
    assert out.shape == q.shape


def test_reference_is_deterministic():
    q, k, v = make_inputs(seed=7)
    a = causal_attention_fp32(q, k, v)
    b = causal_attention_fp32(q, k, v)
    np.testing.assert_array_equal(a, b)


def test_reference_different_value_dimension():
    q, k, v = make_inputs(D=8, Dv=16)
    out = causal_attention_fp32(q, k, v)
    assert out.shape[-1] == 16


def test_future_tokens_do_not_affect_prefix_outputs():
    q, k, v = make_inputs(N=6, D=8)
    out1 = causal_attention_fp32(q, k, v)

    v2 = v.copy()
    # Change only future V positions. Outputs for query rows 0..2 must be unchanged.
    v2[..., 3:, :] = np.float16(123.0)
    out2 = causal_attention_fp32(q, k, v2)

    np.testing.assert_array_equal(out1[..., :3, :], out2[..., :3, :])


def test_future_keys_do_not_affect_prefix_outputs():
    q, k, v = make_inputs(N=6, D=8)
    out1 = causal_attention_fp32(q, k, v)

    k2 = k.copy()
    # Keys 3..5 cannot affect queries 0..2 in causal attention.
    k2[..., 3:, :] = np.float16(-123.0)
    out2 = causal_attention_fp32(q, k2, v)

    np.testing.assert_array_equal(out1[..., :3, :], out2[..., :3, :])


def test_q_changes_only_its_query_row_and_later_rows_may_change():
    q, k, v = make_inputs(N=6, D=8)
    out1 = causal_attention_fp32(q, k, v)

    q2 = q.copy()
    q2[..., 2, :] = np.float16(7.0)
    out2 = causal_attention_fp32(q2, k, v)

    # Earlier queries cannot depend on a changed later query.
    np.testing.assert_array_equal(out1[..., :2, :], out2[..., :2, :])
    assert not np.array_equal(out1[..., 2, :], out2[..., 2, :])


def test_single_token_attention_is_exact_v0():
    rng = np.random.default_rng(9)
    q = rng.standard_normal((2, 3, 1, 8), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((2, 3, 1, 8), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((2, 3, 1, 5), dtype=np.float32).astype(np.float16)

    out = causal_attention_fp32(q, k, v)
    np.testing.assert_array_equal(out, v.astype(np.float32))


@pytest.mark.parametrize("name", ["q", "k", "v"])
def test_nan_and_inf_are_rejected(name):
    q, k, v = make_inputs()
    target = {"q": q, "k": k, "v": v}[name].copy()
    target.flat[0] = np.nan

    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(
            target,
            k,
            v,
        )

    target = {"q": q, "k": k, "v": v}[name].copy()
    target.flat[0] = np.inf
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(
            target,
            k,
            v,
        )


def test_wrong_dtype_rejected():
    q, k, v = make_inputs()
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(q.astype(np.float32), k, v)


def test_wrong_rank_rejected():
    q, k, v = make_inputs()
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(q[0], k, v)


def test_non_contiguous_rejected():
    q, k, v = make_inputs()
    q_nc = np.asfortranarray(q)
    assert not q_nc.flags.c_contiguous
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(q_nc, k, v)


def test_mismatched_batch_heads_rejected():
    q, k, v = make_inputs()
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(q, k[:1], v)


def test_mismatched_sequence_rejected():
    q, k, v = make_inputs(N=4)
    k_bad = np.concatenate([k, k[..., :1, :]], axis=-2)
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(q, k_bad, v)


def test_mismatched_head_dimension_rejected():
    q, k, v = make_inputs(D=8)
    k_bad = np.zeros((1, 2, 4, 16), dtype=np.float16)
    with pytest.raises(AttentionValidationError):
        validate_attention_inputs(q, k_bad, v)


def test_error_helpers():
    expected = np.array([1.0, 2.0], dtype=np.float32)
    actual = np.array([1.001, 1.99], dtype=np.float32)
    assert max_abs_error(actual, expected) > 0
    assert max_rel_error(actual, expected) > 0
    assert passes_mixed_tolerance(
        actual, expected, abs_scale=0.01, rel_scale=0.01
    )
