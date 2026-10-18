import os

import numpy as np
import pytest

from reference import causal_attention_fp32, max_abs_error, max_rel_error

pytestmark = pytest.mark.gpu

if (
    os.environ.get("REQUIRE_A100") == "1"
    or os.environ.get("REQUIRE_CUTILE") == "1"
):
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required for required cuTile edge tests: {exc}", pytrace=False)
    try:
        import cuda.tile as ct  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"cuTile is required for required cuTile edge tests: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")
    pytest.importorskip("cuda.tile")

from cutile_attention import forward


def require_a100():
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"")
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if os.environ.get("REQUIRE_A100") == "1" and "A100" not in name:
        pytest.fail(f"A100 required, found {name}")
    return name


def make_inputs(D=64, N=64, B=1, H=1, seed=9000):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    return q, k, v


def assert_matches_reference(actual, q, k, v):
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    abs_err = max_abs_error(actual, expected)
    rel_err = max_rel_error(actual, expected)
    assert abs_err <= 2e-2 + 2e-2 * float(np.max(np.abs(expected))), (
        f"max_abs={abs_err}, max_rel={rel_err}"
    )


@pytest.mark.parametrize("D", [32, 64, 128])
def test_constant_v_is_preserved_for_all_required_dimensions(D):
    require_a100()
    q, k, _ = make_inputs(D=D, N=64, seed=9001 + D)
    v = np.full_like(q, np.float16(-2.75))

    actual = forward(q, k, v, tile_q=64, tile_k=64)

    np.testing.assert_array_equal(actual, np.full_like(actual, np.float16(-2.75)))


def test_zero_logits_equal_causal_prefix_average():
    require_a100()
    D = 64
    N = 64
    q = np.zeros((1, 1, N, D), dtype=np.float16)
    k = np.zeros_like(q)
    v = np.arange(N * D, dtype=np.float32).reshape(1, 1, N, D).astype(np.float16)

    actual = forward(q, k, v, tile_q=64, tile_k=64)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


def test_tile_configurations_agree_with_each_other():
    require_a100()
    q, k, v = make_inputs(D=64, N=128, seed=9002)
    reference = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    outputs = {}
    for tile_q, tile_k in ((32, 64), (64, 32), (64, 64), (128, 128)):
        outputs[(tile_q, tile_k)] = forward(
            q, k, v, tile_q=tile_q, tile_k=tile_k
        )
        np.testing.assert_allclose(
            outputs[(tile_q, tile_k)], reference, rtol=2e-2, atol=2e-2
        )

    for key, value in outputs.items():
        for other_key, other in outputs.items():
            if key < other_key:
                np.testing.assert_allclose(value, other, rtol=2e-2, atol=2e-2)


def test_future_values_cannot_change_earlier_query_outputs():
    require_a100()
    q, k, v = make_inputs(D=64, N=64, seed=9003)
    before = forward(q, k, v, tile_q=64, tile_k=64)

    v2 = v.copy()
    v2[..., 32:, :] = np.float16(123.0)
    after = forward(q, k, v2, tile_q=64, tile_k=64)

    np.testing.assert_array_equal(before[..., :32, :], after[..., :32, :])


def test_future_keys_cannot_change_earlier_query_outputs():
    require_a100()
    q, k, v = make_inputs(D=64, N=64, seed=9004)
    before = forward(q, k, v, tile_q=64, tile_k=64)

    k2 = k.copy()
    k2[..., 32:, :] = np.float16(-123.0)
    after = forward(q, k2, v, tile_q=64, tile_k=64)

    np.testing.assert_array_equal(before[..., :32, :], after[..., :32, :])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_rejected_before_kernel_execution(bad_value):
    require_a100()
    q, k, v = make_inputs()
    q.flat[0] = np.float16(bad_value)

    with pytest.raises(ValueError, match="NaN|infinity"):
        forward(q, k, v, tile_q=64, tile_k=64)


def test_noncontiguous_input_is_rejected():
    require_a100()
    q, k, v = make_inputs()
    q_nc = np.asfortranarray(q)
    assert not q_nc.flags.c_contiguous

    with pytest.raises(ValueError, match="C-contiguous"):
        forward(q_nc, k, v, tile_q=64, tile_k=64)


@pytest.mark.parametrize("D,Dv", [(16, 16), (64, 32)])
def test_unsupported_or_mismatched_value_shape_is_rejected(D, Dv):
    require_a100()
    q, k, _ = make_inputs(D=D, N=64)
    rng = np.random.default_rng(9005 + D + Dv)
    v = rng.standard_normal((1, 1, 64, Dv), dtype=np.float32).astype(np.float16)

    with pytest.raises(ValueError, match="unsupported|Dv == D"):
        forward(q, k, v, tile_q=64, tile_k=64)


def test_nondivisible_sequence_is_rejected_for_each_tile_dimension():
    require_a100()
    q, k, v = make_inputs(D=64, N=96)

    for tile_q, tile_k in ((64, 64), (128, 64), (64, 128)):
        with pytest.raises(ValueError, match="divisible"):
            forward(q, k, v, tile_q=tile_q, tile_k=tile_k)