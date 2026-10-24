"""Warp-level fused causal attention in ordinary CUDA C++."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


WARP_SIZE = 32
WARPS_PER_BLOCK = 4
BLOCK_THREADS = WARP_SIZE * WARPS_PER_BLOCK


class WarpCudaUnavailableError(RuntimeError):
    """Raised when CuPy/CUDA is unavailable."""


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - GPU environment dependent
        raise WarpCudaUnavailableError("CuPy is required for the warp CUDA baseline") from exc
    return cp


@lru_cache(maxsize=1)
def _kernel():
    cp = _import_cupy()
    source = r"""
    #include <cuda_fp16.h>
    #include <math.h>
    #include <float.h>

    #define FULL_MASK 0xffffffffu
    #define WARP_SIZE 32

    __device__ __forceinline__ float warp_sum(float x) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            x += __shfl_down_sync(FULL_MASK, x, offset);
        }
        return x;
    }

    __device__ __forceinline__ float warp_broadcast_zero(float x) {
        return __shfl_sync(FULL_MASK, x, 0);
    }

    extern "C" __global__
    void causal_attention_warp(
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
        const int lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x >> 5;
        const int qpos = (int)blockIdx.z * 4 + warp_id;
        const int h = (int)blockIdx.y;
        const int b = (int)blockIdx.x;

        if (b >= B || h >= H || qpos >= N) {
            return;
        }

        const long long head_base =
            ((long long)b * H + h) * N * D;
        const long long q_base =
            head_base + (long long)qpos * D;

        float row_max = -FLT_MAX;

        // Stable-softmax pass 1: find the exact maximum logit.
        for (int j = 0; j <= qpos; ++j) {
            const long long k_base = head_base + (long long)j * D;
            float local_dot = 0.0f;

            for (int d = lane; d < D; d += WARP_SIZE) {
                local_dot +=
                    __half2float(Q[q_base + d]) *
                    __half2float(K[k_base + d]);
            }

            float dot = warp_sum(local_dot);
            dot = warp_broadcast_zero(dot * scale);
            row_max = fmaxf(row_max, dot);
        }

        row_max = warp_broadcast_zero(row_max);

        float norm = 0.0f;
        float value_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Stable-softmax pass 2: accumulate normalization and output values.
        for (int j = 0; j <= qpos; ++j) {
            const long long k_base = head_base + (long long)j * D;
            float local_dot = 0.0f;

            for (int d = lane; d < D; d += WARP_SIZE) {
                local_dot +=
                    __half2float(Q[q_base + d]) *
                    __half2float(K[k_base + d]);
            }

            float score = warp_sum(local_dot);
            score = warp_broadcast_zero(score * scale);

            const float weight = expf(score - row_max);
            norm += weight;

            for (int m = 0; m < 4; ++m) {
                const int dv = lane + m * WARP_SIZE;
                if (dv < D) {
                    value_acc[m] +=
                        weight *
                        __half2float(V[head_base + (long long)j * D + dv]);
                }
            }
        }

        for (int m = 0; m < 4; ++m) {
            const int dv = lane + m * WARP_SIZE;
            if (dv < D) {
                O[head_base + (long long)qpos * D + dv] =
                    __float2half(value_acc[m] / norm);
            }
        }
    }
    """
    return cp.RawKernel(
        source,
        "causal_attention_warp",
        options=("--std=c++14", "-arch=sm_80"),
        backend="nvcc",
    )


def _validate_device_inputs(q_gpu, k_gpu, v_gpu):
    cp = _import_cupy()
    for name, x in (("Q", q_gpu), ("K", k_gpu), ("V", v_gpu)):
        if x.dtype != cp.float16:
            raise AttentionValidationError(f"{name} must have dtype float16")
        if x.ndim != 4:
            raise AttentionValidationError(f"{name} must have rank 4")
        if not x.flags.c_contiguous:
            raise AttentionValidationError(f"{name} must be C-contiguous")

    if q_gpu.shape != k_gpu.shape or q_gpu.shape != v_gpu.shape:
        raise AttentionValidationError("warp CUDA self-attention requires matching Q/K/V shapes")
    if q_gpu.shape[-1] not in (32, 64, 128):
        raise AttentionValidationError("warp CUDA attention requires D in {32, 64, 128}")
    if q_gpu.shape[-2] != k_gpu.shape[-2] or q_gpu.shape[-2] != v_gpu.shape[-2]:
        raise AttentionValidationError("Q/K/V sequence lengths must match")


def launch_device(q_gpu, k_gpu, v_gpu):
    cp = _import_cupy()
    _validate_device_inputs(q_gpu, k_gpu, v_gpu)

    b, h, n, d = q_gpu.shape
    out = cp.empty_like(q_gpu)

    _kernel()(
        (b, h, (n + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK),
        (BLOCK_THREADS,),
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


def launch(q: np.ndarray, k: np.ndarray, v: np.ndarray):
    cp = _import_cupy()
    validate_attention_inputs(q, k, v)
    return launch_device(cp.asarray(q), cp.asarray(k), cp.asarray(v))


def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    cp = _import_cupy()
    out_gpu = launch(q, k, v)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)
