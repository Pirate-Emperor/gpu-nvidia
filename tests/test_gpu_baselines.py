import os

import numpy as np
import pytest

from reference import causal_attention_fp32, passes_mixed_tolerance

pytestmark = pytest.mark.gpu


def require_a100(cp):
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"")
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if os.environ.get("REQUIRE_A100") == "1" and "A100" not in name:
        pytest.fail(f"A100 required, found {name}")
    return name


def make_case():
    rng = np.random.default_rng(7001)
    q = rng.standard_normal((1, 1, 64, 32), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((1, 1, 64, 32), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((1, 1, 64, 32), dtype=np.float32).astype(np.float16)
    return q, k, v


def test_simple_cuda_baseline_matches_reference():
    cp = pytest.importorskip("cupy")
    from naive_cuda_attention import forward

    require_a100(cp)
    q, k, v = make_case()
    actual = forward(q, k, v)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    assert np.isfinite(actual).all()
    assert passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2)


def test_unfused_cupy_baseline_matches_reference():
    cp = pytest.importorskip("cupy")
    from baseline import forward

    require_a100(cp)
    q, k, v = make_case()
    actual = forward(q, k, v)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

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
