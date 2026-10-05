"""Deliberately unfused GPU attention baseline for performance comparison."""

from __future__ import annotations

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - GPU environment dependent
        raise RuntimeError("CuPy is required for the GPU baseline") from exc
    return cp


def launch_device(q_gpu, k_gpu, v_gpu):
    """Launch the unfused attention baseline on already resident GPU arrays."""
    cp = _import_cupy()

    scores = cp.matmul(q_gpu, cp.swapaxes(k_gpu, -1, -2))
    scores *= np.float32(1.0 / np.sqrt(q_gpu.shape[-1]))

    n = q_gpu.shape[-2]
    future = cp.triu(cp.ones((n, n), dtype=cp.bool_), k=1)
    scores = cp.where(future[None, None, :, :], -cp.inf, scores)

    row_max = cp.max(scores, axis=-1, keepdims=True)
    exp_scores = cp.exp(scores - row_max)
    probs = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)
    out = cp.matmul(probs, v_gpu)

    return out.astype(cp.float16)


def launch(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    """Copy inputs to the GPU and launch the unfused baseline."""
    cp = _import_cupy()
    validate_attention_inputs(q, k, v)

    q_gpu = cp.asarray(q, dtype=cp.float32)
    k_gpu = cp.asarray(k, dtype=cp.float32)
    v_gpu = cp.asarray(v, dtype=cp.float32)
    return launch_device(q_gpu, k_gpu, v_gpu)


def forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Materialize score/probability matrices and return FP16 output."""
    cp = _import_cupy()
    out_gpu = launch(q, k, v)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)
