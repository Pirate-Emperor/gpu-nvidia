import numpy as np

from reference import causal_attention_fp32, passes_mixed_tolerance


def test_two_token_scalar_causal_attention_has_exact_structure():
    # D=1 makes the expected behavior easy to derive:
    # q=[1, 0], k=[1, 1], V=[10, 20].
    # Row 0 can only attend V0 -> 10 exactly.
    # Row 1 attends both keys equally -> 15.
    q = np.array([[[[1.0], [0.0]]]], dtype=np.float16)
    k = np.array([[[[1.0], [1.0]]]], dtype=np.float16)
    v = np.array([[[[10.0], [20.0]]]], dtype=np.float16)

    out = causal_attention_fp32(q, k, v)

    np.testing.assert_array_equal(out[..., 0, 0], np.array([[10.0]], dtype=np.float32))
    np.testing.assert_allclose(
        out[..., 1, 0],
        np.array([[15.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_future_value_change_cannot_affect_any_earlier_query():
    q = np.array(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float16
    )
    k = np.array(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float16
    )
    v = np.array(
        [[[[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]]], dtype=np.float16
    )

    out1 = causal_attention_fp32(q, k, v)
    v2 = v.copy()
    v2[..., 2, :] = np.array([301.0, -307.0], dtype=np.float16)
    out2 = causal_attention_fp32(q, k, v2)

    np.testing.assert_array_equal(out1[..., :2, :], out2[..., :2, :])
    assert not np.array_equal(out1[..., 2, :], out2[..., 2, :])


def test_future_key_change_cannot_affect_any_earlier_query():
    q = np.array(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float16
    )
    k = np.array(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float16
    )
    v = np.array(
        [[[[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]]], dtype=np.float16
    )

    out1 = causal_attention_fp32(q, k, v)
    k2 = k.copy()
    k2[..., 2, :] = np.array([307.0, -301.0], dtype=np.float16)
    out2 = causal_attention_fp32(q, k2, v)

    np.testing.assert_array_equal(out1[..., :2, :], out2[..., :2, :])
    assert not np.array_equal(out1[..., 2, :], out2[..., 2, :])


def test_query_row_change_does_not_affect_earlier_rows():
    q = np.array(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float16
    )
    k = np.array(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=np.float16
    )
    v = np.array(
        [[[[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]]]], dtype=np.float16
    )

    out1 = causal_attention_fp32(q, k, v)
    q2 = q.copy()
    q2[..., 2, :] = np.array([8.0, -8.0], dtype=np.float16)
    out2 = causal_attention_fp32(q2, k, v)

    np.testing.assert_array_equal(out1[..., :2, :], out2[..., :2, :])
    assert not np.array_equal(out1[..., 2, :], out2[..., 2, :])


def test_online_softmax_reference_handles_large_finite_logits():
    q = np.array([[[[64.0, 0.0], [0.0, 64.0]]]], dtype=np.float16)
    k = np.array([[[[64.0, 0.0], [0.0, 64.0]]]], dtype=np.float16)
    v = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float16)

    out = causal_attention_fp32(q, k, v)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(
        out[..., 0, :], v[..., 0, :].astype(np.float32), atol=1e-5
    )
    np.testing.assert_allclose(
        out[..., 1, :], v[..., 1, :].astype(np.float32), atol=1e-5
    )


def test_error_metric_boundary_is_inclusive():
    expected = np.array([1.0, 100.0], dtype=np.float32)
    allowed = 0.1 + 0.05 * np.abs(expected)
    actual = expected + allowed.astype(np.float32)

    assert passes_mixed_tolerance(
        actual, expected, abs_scale=0.1, rel_scale=0.05
    )


def test_error_metric_rejects_outside_boundary():
    expected = np.array([1.0, 100.0], dtype=np.float32)
    actual = expected + (0.1 + 0.05 * np.abs(expected)) + 1e-3

    assert not passes_mixed_tolerance(
        actual, expected, abs_scale=0.1, rel_scale=0.05
    )
