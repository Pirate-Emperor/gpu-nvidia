"""Independent FP32 reference for causal scaled dot-product attention.

This module deliberately avoids cuTile/CUDA and is used as the correctness oracle.
The reference is written in NumPy so an optimized GPU implementation cannot
accidentally validate itself.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class AttentionValidationError(ValueError):
    """Raised when inputs violate the required attention contract."""


def validate_attention_inputs(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> Tuple[int, int, int, int, int]:
    """Validate the common tensor contract and return B, H, Nq, Nk, D/Dv.

    The mandatory project path is causal self-attention, therefore Nq == Nk and
    the Q/K head dimension must match. Dv may differ from D mathematically, but
    the course shape contract currently requires the supported head dimensions;
    that check belongs to the launch/config layer, not this mathematical oracle.
    """

    for name, x in (("Q", q), ("K", k), ("V", v)):
        if not isinstance(x, np.ndarray):
            raise AttentionValidationError(f"{name} must be a NumPy array")
        if x.ndim != 4:
            raise AttentionValidationError(f"{name} must have rank 4")
        if x.dtype != np.float16:
            raise AttentionValidationError(f"{name} must have dtype float16")
        if not x.flags.c_contiguous:
            raise AttentionValidationError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(x)):
            raise AttentionValidationError(f"{name} contains NaN or infinity")

    bq, hq, nq, d = q.shape
    bk, hk, nk, dk = k.shape
    bv, hv, nvk, dv = v.shape

    if (bq, hq) != (bk, hk) or (bq, hq) != (bv, hv):
        raise AttentionValidationError("Q/K/V batch and head dimensions must match")
    if nq != nk or nk != nvk:
        raise AttentionValidationError(
            "mandatory causal self-attention requires Nq == Nk == V sequence length"
        )
    if d != dk:
        raise AttentionValidationError("Q and K head dimensions must match")

    return bq, hq, nq, nk, dv


def causal_attention_fp32(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *, output_dtype: np.dtype = np.float32
) -> np.ndarray:
    """Compute causal scaled dot-product attention with FP32 intermediates.

    Inputs remain FP16 storage, but all arithmetic after input conversion is
    explicitly performed in FP32. The causal mask is applied before softmax.

    Returns:
        [B, H, N, Dv] output with the requested output dtype.
    """

    validate_attention_inputs(q, k, v)

    q32 = q.astype(np.float32, copy=False)
    k32 = k.astype(np.float32, copy=False)
    v32 = v.astype(np.float32, copy=False)

    d = q.shape[-1]
    scale = np.float32(1.0 / math.sqrt(d))

    # [B,H,N,N]. Explicit matmul keeps this oracle simple and independent of
    # the optimized tiled implementation.
    scores = np.matmul(q32, np.swapaxes(k32, -1, -2))
    scores *= scale

    n = q.shape[-2]
    # True entries are future positions and must never participate in softmax.
    future = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(future[None, None, :, :], np.float32(-np.inf), scores)

    row_max = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - row_max
    exp_scores = np.exp(shifted, dtype=np.float32)
    denom = np.sum(exp_scores, axis=-1, keepdims=True, dtype=np.float32)

    probs = exp_scores / denom
    out = np.matmul(probs, v32)

    result = out.astype(output_dtype, copy=False)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("reference produced NaN or infinity")
    return result


def max_abs_error(actual: np.ndarray, expected: np.ndarray) -> float:
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return float(np.max(diff, initial=0.0))


def max_rel_error(actual: np.ndarray, expected: np.ndarray) -> float:
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    denom = np.maximum(np.abs(e), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(a - e) / denom, initial=0.0))


def passes_mixed_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    abs_scale: float,
    rel_scale: float,
) -> bool:
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    allowed = abs_scale + rel_scale * np.abs(e)
    return bool(np.all(np.abs(a - e) <= allowed))
