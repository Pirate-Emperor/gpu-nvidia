"""Exhaustive small-case validation of the mathematical and online references.

The project GPU kernels must ultimately agree with these references. This module
adds an independent scalar FP64 oracle for small tensors and exhaustively checks
many sequence lengths, head/value dimensions, tilings, and adversarial patterns.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from online_reference import causal_attention_online_fp32
from reference import causal_attention_fp32


def scalar_attention_fp64(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Independent explicit-loop causal attention oracle for small tensors."""
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)

    bsz, heads, n, d = q.shape
    dv = v.shape[-1]
    scale = 1.0 / math.sqrt(d)
    out = np.empty((bsz, heads, n, dv), dtype=np.float64)

    for b in range(bsz):
        for h in range(heads):
            for i in range(n):
                logits = []
                for j in range(i + 1):
                    s = 0.0
                    for x in range(d):
                        s += q64[b, h, i, x] * k64[b, h, j, x]
                    logits.append(s * scale)

                m = max(logits)
                exps = [math.exp(x - m) for x in logits]
                den = math.fsum(exps)

                for z in range(dv):
                    acc = 0.0
                    for j, w in enumerate(exps):
                        acc += w * v64[b, h, j, z]
                    out[b, h, i, z] = acc / den

    return out


def make_inputs(n: int, d: int, dv: int, b: int, h: int, seed: int):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((b, h, n, d), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((b, h, n, d), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((b, h, n, dv), dtype=np.float32).astype(np.float16)
    return q, k, v


@pytest.mark.parametrize("n", range(1, 13))
@pytest.mark.parametrize("d", (1, 2, 3, 4, 8))
@pytest.mark.parametrize("dv", (1, 2, 5))
@pytest.mark.parametrize("b,h", ((1, 1), (1, 2), (2, 1)))
def test_vectorized_reference_matches_independent_scalar(n, d, dv, b, h):
    q, k, v = make_inputs(n, d, dv, b, h, 18000 + 97 * n + 11 * d + dv + b + h)
    vectorized = causal_attention_fp32(q, k, v, output_dtype=np.float64)
    scalar = scalar_attention_fp64(q, k, v)

    assert vectorized.shape == scalar.shape
    assert np.isfinite(vectorized).all()
    np.testing.assert_allclose(vectorized, scalar, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("n", range(1, 17))
@pytest.mark.parametrize("d,dv", ((1, 1), (2, 3), (4, 5), (8, 2)))
def test_online_reference_matches_scalar_for_every_legal_tiling(n, d, dv):
    q, k, v = make_inputs(n, d, dv, 1, 1, 19000 + 31 * n + d + dv)
    scalar = scalar_attention_fp64(q, k, v).astype(np.float32)

    divisors = [x for x in range(1, n + 1) if n % x == 0]
    for tq in divisors:
        for tk in divisors:
            online = causal_attention_online_fp32(
                q, k, v, tile_q=tq, tile_k=tk
            )
            assert np.isfinite(online).all(), f"NaN for n={n}, tq={tq}, tk={tk}"
            np.testing.assert_allclose(
                online,
                scalar,
                rtol=4e-5,
                atol=4e-5,
                err_msg=f"n={n}, tq={tq}, tk={tk}",
            )


@pytest.mark.parametrize("n", (1, 2, 4, 8, 12))
def test_online_reference_handles_entirely_future_tiles(n):
    d = 8
    q, k, v = make_inputs(n, d, 3, 1, 1, 20000 + n)
    scalar = scalar_attention_fp64(q, k, v).astype(np.float32)

    # tq > 1 causes early Q tiles to encounter later, entirely masked K tiles.
    for tq in [x for x in (1, 2, 4, 8) if n % x == 0]:
        for tk in [x for x in (1, 2, 4) if n % x == 0]:
            out = causal_attention_online_fp32(
                q, k, v, tile_q=tq, tile_k=tk
            )
            assert np.isfinite(out).all()
            np.testing.assert_allclose(out, scalar, rtol=4e-5, atol=4e-5)


def scalar_online_kernel_recurrence(q, k, v, tile_k):
    """Mirror the A100 surrogate's per-score online update in FP64."""
    q64 = q.astype(np.float64)
    k64 = k.astype(np.float64)
    v64 = v.astype(np.float64)
    bsz, heads, n, d = q.shape
    dv = v.shape[-1]
    scale = 1.0 / math.sqrt(d)
    out = np.empty((bsz, heads, n, dv), dtype=np.float64)

    for b in range(bsz):
        for h in range(heads):
            for i in range(n):
                m = -math.inf
                z = 0.0
                acc = np.zeros(dv, dtype=np.float64)
                for k0 in range(0, n, tile_k):
                    for j in range(k0, k0 + tile_k):
                        if j > i:
                            continue
                        s = float(np.dot(q64[b, h, i], k64[b, h, j]) * scale)
                        nm = max(m, s)
                        alpha = 0.0 if math.isinf(m) and m < 0 else math.exp(m - nm)
                        w = math.exp(s - nm)
                        z = alpha * z + w
                        acc = alpha * acc + w * v64[b, h, j]
                        m = nm
                out[b, h, i] = acc / z
    return out


@pytest.mark.parametrize("n", range(1, 9))
@pytest.mark.parametrize("d", (32, 64, 128))
@pytest.mark.parametrize("tile_k", (1, 2, 4, 8))
def test_a100_surrogate_recurrence_matches_fp64_oracle(n, d, tile_k):
    if n % tile_k:
        pytest.skip("tile does not divide sequence")
    q, k, v = make_inputs(n, d, d, 1, 1, 20500 + n + d + tile_k)
    scalar = scalar_attention_fp64(q, k, v)
    online = scalar_online_kernel_recurrence(q, k, v, tile_k)
    np.testing.assert_allclose(
        online, scalar, rtol=3e-12, atol=3e-12
    )


@pytest.mark.parametrize("kind", ("zeros", "large", "alternating", "constant_v"))
def test_reference_adversarial_patterns_match_scalar(kind):
    n, d, dv = 8, 8, 5

    if kind == "zeros":
        q = np.zeros((1, 1, n, d), dtype=np.float16)
        k = np.zeros_like(q)
        v = np.arange(n * dv, dtype=np.float32).reshape(1, 1, n, dv).astype(np.float16)
    elif kind == "large":
        q = np.zeros((1, 1, n, d), dtype=np.float16)
        k = np.zeros_like(q)
        for i in range(n):
            q[..., i, i] = np.float16(64.0)
            k[..., i, i] = np.float16(64.0)
        v = np.arange(n * dv, dtype=np.float32).reshape(1, 1, n, dv).astype(np.float16)
    elif kind == "alternating":
        base = np.array([1.0, -1.0], dtype=np.float16)
        q = np.tile(base, (1, 1, n, d // 2))
        k = q.copy()
        v = np.tile(np.arange(dv, dtype=np.float16), (1, 1, n, 1))
    else:
        rng = np.random.default_rng(20123)
        q = rng.standard_normal((1, 1, n, d), dtype=np.float32).astype(np.float16)
        k = rng.standard_normal((1, 1, n, d), dtype=np.float32).astype(np.float16)
        v = np.full((1, 1, n, dv), np.float16(2.75), dtype=np.float16)

    vectorized = causal_attention_fp32(q, k, v, output_dtype=np.float64)
    scalar = scalar_attention_fp64(q, k, v)
    online = causal_attention_online_fp32(q, k, v, tile_q=4, tile_k=2)

    np.testing.assert_allclose(vectorized, scalar, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(online, scalar.astype(np.float32), rtol=4e-5, atol=4e-5)
    assert np.isfinite(vectorized).all()
    assert np.isfinite(online).all()
