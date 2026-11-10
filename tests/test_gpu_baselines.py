import os

import numpy as np
import pytest

from reference import causal_attention_fp32, passes_mixed_tolerance

pytestmark = pytest.mark.gpu


def require_cupy():
    if os.environ.get("REQUIRE_A100") == "1":
        try:
            import cupy as cp
        except ImportError as exc:
            pytest.fail(f"CuPy is required for A100 baseline tests: {exc}", pytrace=False)
        return cp
    return pytest.importorskip("cupy")


def require_a100(cp):
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"")
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if os.environ.get("REQUIRE_A100") == "1" and "A100" not in name:
        pytest.fail(f"A100 required, found {name}")
    return name


@pytest.fixture
def baseline_runtime():
    cp = require_cupy()
    require_a100(cp)
    return cp


def make_case(D=32, seed=7001):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((1, 1, 64, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, 64, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 1, 64, D), dtype=np.float32).astype(np.float16)
    return q, k, v


@pytest.mark.parametrize("D", [32, 64, 128])
def test_simple_cuda_baseline_matches_reference(baseline_runtime, D):
    from naive_cuda_attention import forward

    q, k, v = make_case(D=D, seed=7001 + D)
    actual = forward(q, k, v)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)


@pytest.mark.parametrize("D", [32, 64, 128])
def test_unfused_cupy_baseline_matches_reference(baseline_runtime, D):
    from baseline import forward

    q, k, v = make_case(D=D, seed=7101 + D)
    actual = forward(q, k, v)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    assert actual.shape == expected.shape
    assert actual.dtype == np.float16
    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)


def test_torch_sdpa_is_consistent_when_available():
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA runtime unavailable")

    from torch_reference import forward

    q, k, v = make_case()
    actual = forward(q, k, v)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)
