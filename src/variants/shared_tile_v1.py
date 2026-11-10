"""Shared-memory CUDA baseline: one query row per block."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


BLOCK_THREADS = 128
MAX_TILE_K = 64
MAX_D = 128


class SharedTileCudaUnavailableError(RuntimeError):
    """Raised when CuPy/CUDA is unavailable."""


def _import_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - GPU environment dependent
        raise SharedTileCudaUnavailableError(
            "CuPy is required for the shared-tile CUDA baseline"
        ) from exc
    return cp


@lru_cache(maxsize=8)
def _kernel(tile_k: int):
    cp = _import_cupy()
    if tile_k not in (16, 32, 64):
        raise ValueError("shared-tile kernel supports tile_k in {16, 32, 64}")

    source = r"""
    #include <cuda_fp16.h>
    #include <float.h>
    #include <math.h>

    extern "C" __global__
    void causal_attention_shared_tile(
        const half* __restrict__ Q,
        const half* __restrict__ K,
        const half* __restrict__ V,
        half* __restrict__ O,
        const int B,
        const int H,
        const int N,
        const int D,
        const int TILE_K,
        const float scale)
    {
        __shared__ half q_shared[128];
        __shared__ half k_shared[64 * 128];
        __shared__ half v_shared[64 * 128];
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

        const long long head_base =
            ((long long)b * H + h) * N * D;
        const long long q_base =
            head_base + (long long)qpos * D;

        for (int d = tid; d < D; d += blockDim.x) {
            q_shared[d] = Q[q_base + d];
        }
        __syncthreads();

        float row_max = -FLT_MAX;

        // First pass: stage each K tile and compute the running row maximum.
        for (int k0 = 0; k0 < N; k0 += TILE_K) {
            const int valid = min(TILE_K, N - k0);

            for (int idx = tid; idx < valid * D; idx += blockDim.x) {
                const int j = idx / D;
                const int d = idx - j * D;
                k_shared[j * D + d] =
                    K[head_base + (long long)(k0 + j) * D + d];
            }
            __syncthreads();

            for (int j = 0; j < valid; ++j) {
                float local_dot = 0.0f;
                for (int d = tid; d < D; d += blockDim.x) {
                    local_dot +=
                        __half2float(q_shared[d]) *
                        __half2float(k_shared[j * D + d]);
                }

                reduce_buf[tid] = local_dot;
                __syncthreads();
                for (int stride = 64; stride >= 1; stride >>= 1) {
                    if (tid < stride) {
                        reduce_buf[tid] += reduce_buf[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    shared_score = reduce_buf[0] * scale;
                    if (k0 + j <= qpos) {
                        row_max = fmaxf(row_max, shared_score);
                    }
                }
                __syncthreads();
            }
        }

        if (tid == 0) {
            shared_max = row_max;
        }
        __syncthreads();

        float norm = 0.0f;
        float value_acc = 0.0f;

        // Second pass: stage K/V tiles and accumulate the stable softmax output.
        for (int k0 = 0; k0 < N; k0 += TILE_K) {
            const int valid = min(TILE_K, N - k0);

            for (int idx = tid; idx < valid * D; idx += blockDim.x) {
                const int j = idx / D;
                const int d = idx - j * D;
                k_shared[j * D + d] =
                    K[head_base + (long long)(k0 + j) * D + d];
                v_shared[j * D + d] =
                    V[head_base + (long long)(k0 + j) * D + d];
            }
            __syncthreads();

            for (int j = 0; j < valid; ++j) {
                float local_dot = 0.0f;
                for (int d = tid; d < D; d += blockDim.x) {
                    local_dot +=
                        __half2float(q_shared[d]) *
                        __half2float(k_shared[j * D + d]);
                }

                reduce_buf[tid] = local_dot;
                __syncthreads();
                for (int stride = 64; stride >= 1; stride >>= 1) {
                    if (tid < stride) {
                        reduce_buf[tid] += reduce_buf[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0) {
                    shared_score = reduce_buf[0] * scale;
                }
                __syncthreads();

                if (k0 + j <= qpos) {
                    const float weight = expf(shared_score - shared_max);
                    norm += weight;
                    if (tid < D) {
                        value_acc +=
                            weight * __half2float(v_shared[j * D + tid]);
                    }
                }
                __syncthreads();
            }
        }

        if (tid < D) {
            O[q_base + tid] = __float2half(value_acc / norm);
        }
    }
    """

    return cp.RawKernel(
        source,
        "causal_attention_shared_tile",
        options=("--std=c++14", "-arch=sm_80"),
        backend="nvcc",
    )


def _validate_device_inputs(q_gpu, k_gpu, v_gpu, tile_k: int):
    cp = _import_cupy()

    if tile_k not in (16, 32, 64):
        raise AttentionValidationError("shared-tile key tile must be 32 or 64")
    for name, x in (("Q", q_gpu), ("K", k_gpu), ("V", v_gpu)):
        if x.dtype != cp.float16:
            raise AttentionValidationError(f"{name} must have dtype float16")
        if x.ndim != 4:
            raise AttentionValidationError(f"{name} must have rank 4")
        if not x.flags.c_contiguous:
            raise AttentionValidationError(f"{name} must be C-contiguous")

    if q_gpu.shape != k_gpu.shape or q_gpu.shape != v_gpu.shape:
        raise AttentionValidationError("shared-tile self-attention requires matching Q/K/V shapes")

    if q_gpu.shape[-1] not in (32, 64, 128):
        raise AttentionValidationError("shared-tile attention requires D in {32, 64, 128}")

    if q_gpu.shape[-2] != k_gpu.shape[-2] or q_gpu.shape[-2] != v_gpu.shape[-2]:
        raise AttentionValidationError("Q/K/V sequence lengths must match")

    if q_gpu.shape[-2] % tile_k != 0:
        raise AttentionValidationError(
            f"sequence length {q_gpu.shape[-2]} must be divisible by tile_k {tile_k}"
        )


def launch_device(q_gpu, k_gpu, v_gpu, *, tile_k: int = 64):
    cp = _import_cupy()
    _validate_device_inputs(q_gpu, k_gpu, v_gpu, tile_k)

    b, h, n, d = q_gpu.shape
    out = cp.empty_like(q_gpu)

    _kernel(tile_k)(
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
            tile_k,
            np.float32(1.0 / np.sqrt(d)),
        ),
    )
    return out


def launch(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_k: int = 64,
):
    cp = _import_cupy()
    validate_attention_inputs(q, k, v)
    return launch_device(
        cp.asarray(q),
        cp.asarray(k),
        cp.asarray(v),
        tile_k=tile_k,
    )


def forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_k: int = 64,
) -> np.ndarray:
    cp = _import_cupy()
    out_gpu = launch(q, k, v, tile_k=tile_k)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)
