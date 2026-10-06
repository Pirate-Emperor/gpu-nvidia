"""Simple CUDA baseline: one thread computes one output scalar.

This is intentionally straightforward and inefficient. It uses FP16 inputs,
FP32 score/softmax/value accumulation, and performs two causal key passes per
output scalar. It is a programming-model baseline, not the project target.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


class NaiveCudaUnavailableError(RuntimeError):
    """Raised when CuPy/CUDA is unavailable."""


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - GPU environment dependent
        raise NaiveCudaUnavailableError("CuPy is required for the CUDA baseline") from exc
    return cp


@lru_cache(maxsize=1)
def _kernel():
    cp = _import_cupy()
    source = r"""
    #include <cuda_fp16.h>
    #include <float.h>
    #include <math.h>

    extern "C" __global__
    void causal_attention_naive(
        const half* __restrict__ Q,
        const half* __restrict__ K,
        const half* __restrict__ V,
        half* __restrict__ O,
        const int B,
        const int H,
        const int N,
        const int D,
        const float scale)
    {
        const long long linear =
            (long long)blockIdx.x * blockDim.x + threadIdx.x;
        const long long total = (long long)B * H * N * D;
        if (linear >= total) return;

        long long t = linear;
        const int dv = t % D; t /= D;
        const int qpos = t % N; t /= N;
        const int h = t % H; t /= H;
        const int b = (int)t;

        const long long q_base =
            (((long long)b * H + h) * N + qpos) * D;
        const long long k_base_head =
            ((long long)b * H + h) * N * D;
        const long long v_base_head =
            ((long long)b * H + h) * N * D;

        float max_score = -FLT_MAX;
        for (int j = 0; j <= qpos; ++j) {
            float score = 0.0f;
            const long long k_row = k_base_head + (long long)j * D;
            for (int d = 0; d < D; ++d) {
                score += __half2float(Q[q_base + d]) *
                         __half2float(K[k_row + d]);
            }
            score *= scale;
            max_score = fmaxf(max_score, score);
        }

        float norm = 0.0f;
        float accum = 0.0f;
        for (int j = 0; j <= qpos; ++j) {
            float score = 0.0f;
            const long long k_row = k_base_head + (long long)j * D;
            for (int d = 0; d < D; ++d) {
                score += __half2float(Q[q_base + d]) *
                         __half2float(K[k_row + d]);
            }
            score *= scale;
            const float weight = expf(score - max_score);
            norm += weight;
            const long long v_idx = v_base_head + (long long)j * D + dv;
            accum += weight * __half2float(V[v_idx]);
        }

        O[linear] = __float2half(accum / norm);
    }
    """
    return cp.RawKernel(source, "causal_attention_naive", options=("--std=c++14",))


def launch_device(q_gpu, k_gpu, v_gpu, *, threads: int = 256):
    """Launch the scalar CUDA baseline on resident CuPy arrays."""
    cp = _import_cupy()

    if q_gpu.dtype != cp.float16 or k_gpu.dtype != cp.float16 or v_gpu.dtype != cp.float16:
        raise AttentionValidationError("CUDA baseline requires FP16 input tensors")
    if q_gpu.ndim != 4 or k_gpu.ndim != 4 or v_gpu.ndim != 4:
        raise AttentionValidationError("CUDA baseline requires rank-4 tensors")
    if q_gpu.shape != k_gpu.shape or q_gpu.shape != v_gpu.shape:
        raise AttentionValidationError("self-attention baseline requires matching Q/K/V shapes")
    if q_gpu.shape[-1] not in (32, 64, 128):
        raise AttentionValidationError("CUDA baseline requires D in {32, 64, 128}")
    if q_gpu.shape[-2] != k_gpu.shape[-2] or q_gpu.shape[-2] != v_gpu.shape[-2]:
        raise AttentionValidationError("Q/K/V sequence lengths must match")
    if threads <= 0 or threads > 1024:
        raise AttentionValidationError("threads must be in 1..1024")

    b, h, n, d = q_gpu.shape
    out = cp.empty_like(q_gpu)
    total = b * h * n * d
    grid = ((total + threads - 1) // threads,)
    block = (threads,)

    _kernel()(
        grid,
        block,
        (
            q_gpu,
            k_gpu,
            v_gpu,
            out,
            b,
            h,
            n,
            d,
            np.float32(1.0 / np.sqrt(d)),
        ),
    )
    return out


def launch(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, threads: int = 256):
    """Copy inputs to the GPU and launch the scalar CUDA baseline."""
    cp = _import_cupy()
    validate_attention_inputs(q, k, v)
    return launch_device(cp.asarray(q), cp.asarray(k), cp.asarray(v), threads=threads)


def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, threads: int = 256) -> np.ndarray:
    """Compute causal attention with the scalar CUDA baseline."""
    cp = _import_cupy()
    out = launch(q, k, v, threads=threads)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out)
