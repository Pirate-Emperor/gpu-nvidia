"""Custom fused CUDA baseline without cuTile.

This kernel is deliberately written as ordinary CUDA C++ and compiled by CuPy.
One CUDA block owns one causal query row. Threads cooperate to compute each
Q·K score, then each active output thread streams the corresponding V component
through the stable online-softmax recurrence.

This is NOT the course cuTile kernel. It is an independent algorithmic/low-level
reference implementation that lets us validate the fused attention mathematics
and memory/synchronization strategy before the cuTile runtime is available.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


BLOCK_THREADS = 128


class CustomCudaUnavailableError(RuntimeError):
    """Raised when CuPy/CUDA is unavailable."""


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - GPU environment dependent
        raise CustomCudaUnavailableError(
            "CuPy is required for the custom fused CUDA baseline"
        ) from exc
    return cp


@lru_cache(maxsize=1)
def _kernel():
    cp = _import_cupy()
    source = r"""
    #include <cuda_fp16.h>
    #include <float.h>
    #include <math.h>

    __device__ __forceinline__ void reduce_sum_128(float* x) {
        for (int stride = 64; stride >= 1; stride >>= 1) {
            __syncthreads();
            if (threadIdx.x < stride) {
                x[threadIdx.x] += x[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    extern "C" __global__
    void causal_attention_fused_rowwise(
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
        __shared__ float reduce_buf[128];
        __shared__ float shared_score;
        __shared__ float shared_max;

        const int tid = threadIdx.x;
        const int qpos = (int)blockIdx.z;
        const int h = (int)blockIdx.y;
        const int b = (int)blockIdx.x;

        if (b >= B || h >= H || qpos >= N) {
            return;
        }

        const long long q_base =
            (((long long)b * H + h) * N + qpos) * D;
        const long long head_base =
            ((long long)b * H + h) * N * D;

        float row_max = -FLT_MAX;

        // Pass 1: exact same score stream as a stable two-pass softmax.
        for (int j = 0; j <= qpos; ++j) {
            const long long k_base = head_base + (long long)j * D;
            float local_dot = 0.0f;

            for (int d = tid; d < D; d += blockDim.x) {
                local_dot +=
                    __half2float(Q[q_base + d]) *
                    __half2float(K[k_base + d]);
            }

            reduce_buf[tid] = local_dot;
            reduce_sum_128(reduce_buf);

            if (tid == 0) {
                shared_score = reduce_buf[0] * scale;
                row_max = fmaxf(row_max, shared_score);
            }
            __syncthreads();

            if (tid == 0) {
                shared_max = row_max;
            }
            __syncthreads();
            row_max = shared_max;
        }

        float norm = 0.0f;
        float value_acc = 0.0f;

        // Pass 2: streaming normalization and one output dimension per thread.
        const bool active_output = (tid < D);
        const long long out_index =
            head_base + (long long)qpos * D + tid;

        for (int j = 0; j <= qpos; ++j) {
            const long long k_base = head_base + (long long)j * D;
            float local_dot = 0.0f;

            for (int d = tid; d < D; d += blockDim.x) {
                local_dot +=
                    __half2float(Q[q_base + d]) *
                    __half2float(K[k_base + d]);
            }

            reduce_buf[tid] = local_dot;
            reduce_sum_128(reduce_buf);

            if (tid == 0) {
                shared_score = reduce_buf[0] * scale;
            }
            __syncthreads();

            const float weight = expf(shared_score - shared_max);
            if (active_output) {
                norm += weight;
                const long long v_index =
                    head_base + (long long)j * D + tid;
                value_acc += weight * __half2float(V[v_index]);
            }

            __syncthreads();
        }

        if (active_output) {
            O[out_index] = __float2half(value_acc / norm);
        }
    }
    """
    return cp.RawKernel(
        source,
        "causal_attention_fused_rowwise",
        options=("--std=c++14",),
    )


def _validate_device_inputs(q_gpu, k_gpu, v_gpu):
    cp = _import_cupy()

    for name, x in (("Q", q_gpu), ("K", k_gpu), ("V", v_gpu)):
        if x.dtype != cp.float16:
            raise AttentionValidationError(
                f"{name} must have dtype float16 for custom CUDA attention"
            )
        if x.ndim != 4:
            raise AttentionValidationError(
                f"{name} must have rank 4 for custom CUDA attention"
            )
        if not x.flags.c_contiguous:
            raise AttentionValidationError(
                f"{name} must be C-contiguous for custom CUDA attention"
            )

    if q_gpu.shape != k_gpu.shape or q_gpu.shape != v_gpu.shape:
        raise AttentionValidationError(
            "custom CUDA self-attention requires matching Q/K/V shapes"
        )

    if q_gpu.shape[-1] not in (32, 64, 128):
        raise AttentionValidationError("custom CUDA attention requires D in {32, 64, 128}")

    if q_gpu.shape[-2] != k_gpu.shape[-2] or q_gpu.shape[-2] != v_gpu.shape[-2]:
        raise AttentionValidationError("Q/K/V sequence lengths must match")


def launch_device(q_gpu, k_gpu, v_gpu):
    """Launch the custom row-wise fused CUDA kernel on resident arrays."""
    cp = _import_cupy()
    _validate_device_inputs(q_gpu, k_gpu, v_gpu)

    b, h, n, d = q_gpu.shape
    out = cp.empty_like(q_gpu)

    _kernel()(
        (b, h, n),
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
    """Copy host inputs to GPU and launch the custom fused CUDA kernel."""
    cp = _import_cupy()
    validate_attention_inputs(q, k, v)
    return launch_device(cp.asarray(q), cp.asarray(k), cp.asarray(v))


def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute causal attention using the custom fused CUDA kernel."""
    cp = _import_cupy()
    out_gpu = launch(q, k, v)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)
