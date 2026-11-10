import os

import numpy as np
import pytest

from cases import MANDATORY_CASES
from extended_attention import causal_attention_gqa_fp32, causal_attention_packed_fp32
from reference import passes_mixed_tolerance

if os.environ.get("REQUIRE_CUTILE") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy required for optional GQA/packed GPU gates: {exc}", pytrace=False)
    try:
        import cuda.tile as ct
    except ImportError as exc:
        pytest.fail(f"cuTile required for optional GQA/packed GPU gates: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")
    ct = pytest.importorskip("cuda.tile")

from cutile_attention import forward_gqa, forward_packed_varlen


def make_inputs(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32).astype(np.float16)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_cutile_gqa_mqa_matches_independent_reference(kv_heads):
    q = make_inputs((1, 4, 64, 32), 1100 + kv_heads)
    k = make_inputs((1, kv_heads, 64, 32), 1200 + kv_heads)
    v = make_inputs((1, kv_heads, 64, 32), 1300 + kv_heads)
    actual = forward_gqa(q, k, v, tile_q=64, tile_k=64)
    expected = causal_attention_gqa_fp32(q, k, v, output_dtype=np.float16)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)


def test_cutile_packed_varlen_has_no_cross_sequence_leakage():
    q = make_inputs((1, 2, 192, 32), 1401)
    k = make_inputs((1, 2, 192, 32), 1402)
    v = make_inputs((1, 2, 192, 32), 1403)
    offsets = np.asarray([0, 64, 192], dtype=np.int32)
    actual = forward_packed_varlen(q, k, v, offsets, tile_q=64, tile_k=64)
    expected = causal_attention_packed_fp32(q, k, v, offsets, output_dtype=np.float16)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)

    changed_v = v.copy()
    changed_v[..., 64:, :] = np.float16(123.0)
    changed = forward_packed_varlen(q, k, changed_v, offsets, tile_q=64, tile_k=64)
    np.testing.assert_array_equal(actual[..., :64, :], changed[..., :64, :])
