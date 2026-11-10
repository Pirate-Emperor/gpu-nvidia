import numpy as np
import pytest

from online_reference import causal_attention_online_fp32
from reference import causal_attention_fp32, max_abs_error, max_rel_error


def make_inputs(B, H, N, D, Dv, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((B, H, N, D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((B, H, N, Dv), dtype=np.float32).astype(np.float16)
    return q, k, v


@pytest.mark.parametrize(
    "B,H,N,D,Dv,tq,tk,seed",
    [
        (1, 1, 4, 8, 8, 1, 1, 1),
        (1, 1, 8, 8, 5, 2, 4, 2),
        (1, 2, 16, 16, 16, 4, 8, 3),
        (2, 2, 32, 32, 16, 8, 4, 4),
        (1, 4, 64, 64, 64, 16, 16, 5),
        (1, 2, 128, 128, 128, 32, 64, 6),
    ],
)
def test_online_recurrence_matches_full_reference(
    B, H, N, D, Dv, tq, tk, seed
):
    q, k, v = make_inputs(B, H, N, D, Dv, seed)

    full = causal_attention_fp32(q, k, v)
    online = causal_attention_online_fp32(
        q, k, v, tile_q=tq, tile_k=tk
    )

    assert online.shape == full.shape
    assert np.isfinite(online).all()

    # Both references use FP32 arithmetic. The online recurrence changes
    # summation order, so compare numerically rather than bitwise.
    abs_err = max_abs_error(online, full)
    rel_err = max_rel_error(online, full)
    assert np.allclose(online, full, rtol=2e-5, atol=2e-5), (
        f"online recurrence mismatch: max_abs={abs_err}, max_rel={rel_err}"
    )


@pytest.mark.parametrize("tile_q,tile_k", [(2, 2), (4, 8), (8, 4), (16, 16)])
def test_online_recurrence_is_invariant_to_legal_tiling(tile_q, tile_k):
    q, k, v = make_inputs(1, 1, 32, 32, 17, tile_q + tile_k)
    baseline = causal_attention_online_fp32(
        q, k, v, tile_q=tile_q, tile_k=tile_k
    )
    full = causal_attention_fp32(q, k, v)
    np.testing.assert_allclose(baseline, full, rtol=2e-5, atol=2e-5)


def test_online_large_logits_remain_finite():
    rng = np.random.default_rng(77)
    q = (
        rng.standard_normal((1, 1, 64, 64), dtype=np.float32) * 32.0
    ).astype(np.float16)
    k = (
        rng.standard_normal((1, 1, 64, 64), dtype=np.float32) * 32.0
    ).astype(np.float16)
    v = rng.standard_normal((1, 1, 64, 64), dtype=np.float32).astype(np.float16)

    out = causal_attention_online_fp32(q, k, v, tile_q=16, tile_k=16)
    assert np.isfinite(out).all()
