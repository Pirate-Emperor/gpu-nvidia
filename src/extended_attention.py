"""Independent CPU references for optional attention extensions.

These routines use NumPy only, never call cuTile or CUDA, and are suitable as
oracles for grouped-query/multi-query and packed variable-length semantics.
"""
from __future__ import annotations

import numpy as np

from reference import AttentionValidationError, causal_attention_fp32


def causal_attention_gqa_fp32(q, k, v, *, output_dtype=np.float32):
    """Independent causal GQA/MQA reference.

    Q has Hq heads and K/V have Hkv heads. For query head h, KV head
    floor(h / (Hq/Hkv)) is used. Returned tensor has shape [B,Hq,N,Dv].
    """
    for name, x in (("Q", q), ("K", k), ("V", v)):
        if not isinstance(x, np.ndarray) or x.ndim != 4:
            raise AttentionValidationError(f"{name} must be a rank-4 NumPy array")
        if x.dtype != np.float16:
            raise AttentionValidationError(f"{name} must have dtype float16")
        if not x.flags.c_contiguous:
            raise AttentionValidationError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(x)):
            raise AttentionValidationError(f"{name} contains NaN or infinity")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise AttentionValidationError("Q/K/V batch dimensions must match")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise AttentionValidationError("Q/K/V sequence lengths must match")
    if k.shape[1] != v.shape[1] or k.shape[1] < 1:
        raise AttentionValidationError("K and V must have the same positive KV-head count")
    if q.shape[1] % k.shape[1] != 0:
        raise AttentionValidationError("query-head count must be divisible by KV-head count")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise AttentionValidationError("Q/K/V head dimensions must match")
    if q.shape[-1] not in (32, 64, 128):
        raise AttentionValidationError("head dimension must be one of 32, 64, 128")

    groups = q.shape[1] // k.shape[1]
    output = np.empty(q.shape, dtype=output_dtype)
    for hq in range(q.shape[1]):
        hkv = hq // groups
        q_one = np.ascontiguousarray(q[:, hq:hq + 1])
        k_one = np.ascontiguousarray(k[:, hkv:hkv + 1])
        v_one = np.ascontiguousarray(v[:, hkv:hkv + 1])
        output[:, hq:hq + 1] = causal_attention_fp32(
            q_one, k_one, v_one, output_dtype=output_dtype
        )
    return output


def causal_attention_packed_fp32(q, k, v, cu_seqlens, *, output_dtype=np.float32):
    """Independent packed-varlen reference, resetting causality per segment.

    Packed tensors have shape [1,H,total_tokens,D]. Every segment is evaluated
    independently and then written into its original token interval. The
    offsets are exclusive cumulative offsets including zero and total_tokens.
    """
    offsets = np.asarray(cu_seqlens)
    if offsets.ndim != 1 or offsets.size < 2 or offsets.dtype.kind not in "iu":
        raise AttentionValidationError("cu_seqlens must be a 1-D integer offset array of length >= 2")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise AttentionValidationError("packed Q/K/V must have rank 4")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise AttentionValidationError("packed varlen currently expects batch dimension 1")
    if q.shape[-2] != k.shape[-2] or q.shape[-2] != v.shape[-2]:
        raise AttentionValidationError("packed Q/K/V must have the same total token count")
    if int(offsets[0]) != 0 or int(offsets[-1]) != q.shape[-2]:
        raise AttentionValidationError("cu_seqlens must start at 0 and end at total token count")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise AttentionValidationError("cu_seqlens offsets must be strictly increasing")
    outputs = []
    for start_raw, end_raw in zip(offsets[:-1], offsets[1:]):
        start, end = int(start_raw), int(end_raw)
        outputs.append(causal_attention_fp32(
            np.ascontiguousarray(q[..., start:end, :]),
            np.ascontiguousarray(k[..., start:end, :]),
            np.ascontiguousarray(v[..., start:end, :]),
            output_dtype=output_dtype,
        ))
    return np.concatenate(outputs, axis=-2)
