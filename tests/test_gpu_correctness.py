import os

import numpy as np
import pytest

from cases import MANDATORY_CASES
from reference import (
    causal_attention_fp32,
    max_abs_error,
    max_rel_error,
    passes_mixed_tolerance,
)

if (
    os.environ.get("REQUIRE_A100") == "1"
    or os.environ.get("REQUIRE_CUTILE") == "1"
):
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required for required cuTile GPU tests: {exc}", pytrace=False)
    try:
        import cuda.tile as ct
    except ImportError as exc:
        pytest.fail(f"cuTile is required for required cuTile GPU tests: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")
    ct = pytest.importorskip("cuda.tile")

from cutile_attention import forward

pytestmark = pytest.mark.gpu


def require_a100():
    names = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = names.get("name", b"")
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if os.environ.get("REQUIRE_A100") == "1":
        if "A100" not in name:
            pytest.fail(f"A100 required, found {name}")
    return name


def make_inputs(case, mode="normal"):
    rng = np.random.default_rng(case.seed)
    if mode == "large_logits":
        scale = 64.0
    elif mode == "small":
        scale = 1e-3
    else:
        scale = 1.0

    q = (
        rng.standard_normal(
            (case.B, case.H, case.N, case.D),
            dtype=np.float32,
        )
        * scale
    ).astype(np.float16)
    k = (
        rng.standard_normal(
            (case.B, case.H, case.N, case.D),
            dtype=np.float32,
        )
        * scale
    ).astype(np.float16)
    v = (
        rng.standard_normal(
            (case.B, case.H, case.N, case.Dv),
            dtype=np.float32,
        )
        * scale
    ).astype(np.float16)
    return q, k, v


# These two are the mandatory course comparison points.
# 128x128 remains an exploratory/resource-stress configuration below.
TILE_CONFIGS = ((64, 64), (32, 64))
EXPLORATORY_TILE_CONFIGS = ((64, 32), (128, 128))

GPU_CORRECTNESS_CONFIGS = tuple(
    (case, tile_q, tile_k)
    for case in MANDATORY_CASES
    for tile_q, tile_k in TILE_CONFIGS
    if case.N % tile_q == 0 and case.N % tile_k == 0
)


def gpu_config_id(config):
    case, tile_q, tile_k = config
    return f"{case.name}-tile{tile_q}x{tile_k}"


@pytest.mark.parametrize(
    "case,tile_q,tile_k",
    GPU_CORRECTNESS_CONFIGS,
    # pytest's ids callable is invoked per scalar argument, not per full tuple.
    ids=[gpu_config_id(config) for config in GPU_CORRECTNESS_CONFIGS],
)
def test_cutile_matches_independent_reference(case, tile_q, tile_k):
    require_a100()
    q, k, v = make_inputs(case)

    actual = forward(q, k, v, tile_q=tile_q, tile_k=tile_k)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()

    abs_err = max_abs_error(actual, expected)
    rel_err = max_rel_error(actual, expected)

    assert passes_mixed_tolerance(
        actual,
        expected,
        abs_scale=2e-2,
        rel_scale=2e-2,
    ), f"max_abs={abs_err}, max_rel={rel_err}"


@pytest.mark.parametrize(
    "tile_q,tile_k",
    EXPLORATORY_TILE_CONFIGS,
    # Provide one ID per pair; pytest id functions receive scalar values.
    ids=[f"tile{tile_q}x{tile_k}" for tile_q, tile_k in EXPLORATORY_TILE_CONFIGS],
)
def test_cutile_exploratory_tile_configuration(tile_q, tile_k):
    require_a100()
    case = MANDATORY_CASES[2]
    if case.N % tile_q or case.N % tile_k:
        pytest.skip("sequence length is not divisible by exploratory tile")
    q, k, v = make_inputs(case)
    actual = forward(q, k, v, tile_q=tile_q, tile_k=tile_k)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
    assert passes_mixed_tolerance(
        actual, expected, abs_scale=2e-2, rel_scale=2e-2
    )


@pytest.mark.parametrize("mode", ["small", "large_logits"])
def test_cutile_numerical_stability(mode):
    require_a100()
    case = MANDATORY_CASES[0]
    q, k, v = make_inputs(case, mode=mode)
    actual = forward(q, k, v, tile_q=64, tile_k=64)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(
        actual,
        expected,
        abs_scale=2e-2,
        rel_scale=2e-2,
    )


def test_cutile_causal_future_value_invariance():
    require_a100()
    q, k, v = make_inputs(MANDATORY_CASES[1])
    out1 = forward(q, k, v, tile_q=64, tile_k=64)

    v2 = v.copy()
    v2[..., 96:, :] = np.float16(123.0)
    out2 = forward(q, k, v2, tile_q=64, tile_k=64)

    np.testing.assert_array_equal(out1[..., :96, :], out2[..., :96, :])


def test_cutile_rejects_nondivisible_sequence():
    require_a100()
    q, k, v = make_inputs(MANDATORY_CASES[0])
    q = np.concatenate([q, np.zeros_like(q[..., :1, :])], axis=-2)
    k = np.concatenate([k, np.zeros_like(k[..., :1, :])], axis=-2)
    v = np.concatenate([v, np.zeros_like(v[..., :1, :])], axis=-2)

    with pytest.raises(ValueError, match="sequence length"):
        forward(q, k, v, tile_q=64, tile_k=64)
