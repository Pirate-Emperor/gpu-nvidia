"""Shared-memory tiled fused CUDA attention (promoted v3).

This is the closest ordinary-CUDA analogue to the tile-oriented project kernel:
one block owns a small query tile; four QK scores are computed per loop iteration., each warp owns one query row, K/V are staged
in shared-memory tiles, and a fused online-softmax recurrence is performed
without materializing scores or probabilities in global memory.

It is intentionally independent of cuTile.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


BLOCK_THREADS = 128
WARPS_PER_BLOCK = 4
WARP_SIZE = 32
TILE_Q = WARPS_PER_BLOCK
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
    #include <math.h>

    #define WARP_SIZE 32
    #define FULL_MASK 0xffffffffu
    #define TILE_Q 4

    __device__ __forceinline__ float warp_sum(float x) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            x += __shfl_down_sync(FULL_MASK, x, offset);
        }
        return x;
    }

    __device__ __forceinline__ float warp_broadcast(float x) {
        return __shfl_sync(FULL_MASK, x, 0);
    }

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
        __shared__ half q_shared[TILE_Q * 128];
        __shared__ half k_shared[64 * 128];
        __shared__ half v_shared[64 * 128];

        const int tid = threadIdx.x;
        const int lane = tid & (WARP_SIZE - 1);
        const int warp_id = tid >> 5;

        const int qbase = (int)blockIdx.z * TILE_Q;
        const int qpos = qbase + warp_id;
        const int h = (int)blockIdx.y;
        const int b = (int)blockIdx.x;

        if (b >= B || h >= H || warp_id >= TILE_Q) {
            return;
        }

        const long long head_base =
            ((long long)b * H + h) * N * D;
        const long long q_base =
            head_base + (long long)qpos * D;

        // Stage a small query tile once. Each warp owns one query row.
        for (int idx = tid; idx < TILE_Q * D; idx += blockDim.x) {
            const int qr = idx / D;
            const int d = idx - qr * D;
            const int gq = qbase + qr;
            q_shared[qr * D + d] =
                (gq < N) ? Q[head_base + (long long)gq * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        // Each lane owns up to four output dimensions.
        float run_max = -INFINITY;
        float run_norm = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int k0 = 0; k0 < N; k0 += TILE_K) {
            for (int idx = tid; idx < TILE_K * D; idx += blockDim.x) {
                const int j = idx / D;
                const int d = idx - j * D;
                k_shared[j * D + d] =
                    K[head_base + (long long)(k0 + j) * D + d];
                v_shared[j * D + d] =
                    V[head_base + (long long)(k0 + j) * D + d];
            }
            __syncthreads();

            for (int jj = 0; jj < TILE_K; jj += 4) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int d = lane; d < D; d += WARP_SIZE) {
                    const float qd =
                        __half2float(q_shared[(qpos - qbase) * D + d]);
                    s0 += qd * __half2float(k_shared[(jj + 0) * D + d]);
                    s1 += qd * __half2float(k_shared[(jj + 1) * D + d]);
                    s2 += qd * __half2float(k_shared[(jj + 2) * D + d]);
                    s3 += qd * __half2float(k_shared[(jj + 3) * D + d]);
                }

                float sc0 = warp_broadcast(warp_sum(s0) * scale);
                float sc1 = warp_broadcast(warp_sum(s1) * scale);
                float sc2 = warp_broadcast(warp_sum(s2) * scale);
                float sc3 = warp_broadcast(warp_sum(s3) * scale);
                const float scores[4] = {sc0, sc1, sc2, sc3};

                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int j = jj + u;
                    const int kp = k0 + j;
                    if (kp <= qpos) {
                        const float score = scores[u];
                        const float new_max = fmaxf(run_max, score);
                        const float rescale =
                            (run_max == -INFINITY) ? 0.0f : expf(run_max - new_max);
                        const float weight = expf(score - new_max);

                        run_norm = rescale * run_norm + weight;

                        #pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            const int dv = lane + m * WARP_SIZE;
                            if (dv < D) {
                                out_acc[m] =
                                    rescale * out_acc[m]
                                    + weight * __half2float(v_shared[j * D + dv]);
                            }
                        }
                        run_max = new_max;
                    }
                }
            }
            __syncthreads();
        }

        if (qpos < N) {
            for (int m = 0; m < 4; ++m) {
                const int dv = lane + m * WARP_SIZE;
                if (dv < D) {
                    O[q_base + dv] = __float2half(out_acc[m] / run_norm);
                }
            }
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
    if q_gpu.shape[-2] % TILE_Q != 0:
        raise AttentionValidationError(
            f"sequence length {q_gpu.shape[-2]} must be divisible by query tile {TILE_Q}"
        )


def launch_device(q_gpu, k_gpu, v_gpu, *, tile_k: int = 16):
    cp = _import_cupy()
    _validate_device_inputs(q_gpu, k_gpu, v_gpu, tile_k)

    b, h, n, d = q_gpu.shape
    out = cp.empty_like(q_gpu)

    _kernel(tile_k)(
        (b, h, n // TILE_Q),
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
