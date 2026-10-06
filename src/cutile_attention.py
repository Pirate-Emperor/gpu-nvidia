"""cuTile implementation of fused causal scaled dot-product attention."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from contract import TileConfig, validate_launch_contract
from reference import AttentionValidationError


OUTPUT_DTYPE = np.float16


class CuTileUnavailableError(RuntimeError):
    """Raised when the cuTile/CuPy runtime is unavailable."""


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _import_runtime() -> tuple[Any, Any]:
    try:
        import cupy as cp
        import cuda.tile as ct
    except Exception as exc:  # pragma: no cover - depends on GPU environment
        raise CuTileUnavailableError(
            "cuTile/CuPy runtime is unavailable; run on the A100 project node"
        ) from exc
    return cp, ct


@lru_cache(maxsize=64)
def _build_kernel(
    tile_q: int,
    tile_k: int,
    head_dim: int,
    value_dim: int,
):
    cp, ct = _import_runtime()

    @ct.kernel
    def kernel(
        Q,
        K,
        V,
        O,
        TQ: ct.Constant[int],
        TK: ct.Constant[int],
        D: ct.Constant[int],
        DV: ct.Constant[int],
        SCALE: ct.Constant[float],
    ):
        # One block owns one (batch, head, query-tile) work item and streams
        # key/value tiles through the online-softmax state.
        batch_id = ct.bid(0)
        head_id = ct.bid(1)
        query_block = ct.bid(2)

        q4 = ct.load(
            Q,
            index=(batch_id, head_id, query_block, 0),
            shape=(1, 1, TQ, D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        q = ct.reshape(q4, (TQ, D))

        running_max = ct.full((TQ, 1), float("-inf"), dtype=ct.float32)
        running_norm = ct.zeros((TQ, 1), dtype=ct.float32)
        running_out = ct.zeros((TQ, DV), dtype=ct.float32)

        q_pos = ct.reshape(ct.arange(TQ, dtype=ct.int32), (TQ, 1))
        q_pos = q_pos + query_block * TQ

        num_k_tiles = ct.num_tiles(
            K,
            2,
            shape=(1, 1, TK, D),
        )

        for key_block in range(num_k_tiles):
            k4 = ct.load(
                K,
                index=(batch_id, head_id, key_block, 0),
                shape=(1, 1, TK, D),
                padding_mode=ct.PaddingMode.ZERO,
            )
            v4 = ct.load(
                V,
                index=(batch_id, head_id, key_block, 0),
                shape=(1, 1, TK, DV),
                padding_mode=ct.PaddingMode.ZERO,
            )

            k = ct.reshape(k4, (TK, D))
            v = ct.reshape(v4, (TK, DV))

            kt = ct.transpose(k)

            # FP16 operands with an explicit FP32 accumulator satisfy the
            # mandatory precision requirement.
            score_acc = ct.zeros((TQ, TK), dtype=ct.float32)
            scores = ct.mma(q, kt, score_acc)
            scores = scores * SCALE

            k_pos = ct.reshape(ct.arange(TK, dtype=ct.int32), (1, TK))
            k_pos = k_pos + key_block * TK

            future = k_pos > q_pos
            scores = ct.where(
                future,
                float("-inf"),
                scores,
            )

            tile_max = ct.max(scores, axis=1, keepdims=True)
            new_max = ct.maximum(running_max, tile_max)

            rescale = ct.exp(running_max - new_max)
            probabilities = ct.exp(scores - new_max)

            new_norm = rescale * running_norm + ct.sum(
                probabilities,
                axis=1,
                keepdims=True,
            )

            # The probabilities are FP32 after stable softmax. Cast V to
            # FP32 so the MMA operands have a common dtype and retain FP32
            # accumulation.
            v32 = v.astype(ct.float32)
            previous_out = rescale * running_out
            running_out = ct.mma(probabilities, v32, previous_out)
            running_norm = new_norm
            running_max = new_max

        output = running_out / running_norm
        output4 = ct.reshape(output, (1, 1, TQ, DV))
        ct.store(
            O,
            index=(batch_id, head_id, query_block, 0),
            tile=output4.astype(O.dtype),
        )

    return kernel


def _validate_cutile_config(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    tile_q: int,
    tile_k: int,
) -> TileConfig:
    tile = TileConfig(query=tile_q, key=tile_k)
    validate_launch_contract(q, k, v, tile)

    head_dim = q.shape[-1]
    value_dim = v.shape[-1]

    if head_dim not in (32, 64, 128):
        raise AttentionValidationError(
            f"head dimension D={head_dim} is unsupported; expected one of 32, 64, 128"
        )
    if value_dim != head_dim:
        raise AttentionValidationError(
            f"self-attention requires Dv == D; got D={head_dim}, Dv={value_dim}"
        )

    for name, value in (
        ("query tile", tile_q),
        ("key tile", tile_k),
        ("head dimension", head_dim),
        ("value dimension", value_dim),
    ):
        if not _is_power_of_two(value):
            raise AttentionValidationError(
                f"{name} must be a power of two for cuTile"
            )

    return tile


def launch_device(
    q_gpu,
    k_gpu,
    v_gpu,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
):
    """Launch fused causal attention on already resident GPU arrays."""
    cp, ct = _import_runtime()

    batch, heads, seq_len, head_dim = q_gpu.shape
    value_dim = v_gpu.shape[-1]
    if head_dim not in (32, 64, 128):
        raise AttentionValidationError(
            f"head dimension D={head_dim} is unsupported; expected one of 32, 64, 128"
        )
    if value_dim != head_dim:
        raise AttentionValidationError(
            f"self-attention requires Dv == D; got D={head_dim}, Dv={value_dim}"
        )

    tile = TileConfig(query=tile_q, key=tile_k)
    if seq_len % tile.query != 0 or seq_len % tile.key != 0:
        raise AttentionValidationError(
            f"sequence length {seq_len} must be divisible by query/key tiles "
            f"{tile.query}/{tile.key}"
        )
    for name, value in (
        ("query tile", tile_q),
        ("key tile", tile_k),
        ("head dimension", head_dim),
        ("value dimension", value_dim),
    ):
        if not _is_power_of_two(value):
            raise AttentionValidationError(
                f"{name} must be a power of two for cuTile"
            )

    out_gpu = cp.empty(
        (batch, heads, seq_len, value_dim),
        dtype=cp.float16,
    )

    kernel = _build_kernel(tile.query, tile.key, head_dim, value_dim)
    grid = (
        batch,
        heads,
        seq_len // tile.query,
    )
    scale = 1.0 / np.sqrt(head_dim)

    kernel_args = (
        q_gpu,
        k_gpu,
        v_gpu,
        out_gpu,
        tile.query,
        tile.key,
        head_dim,
        value_dim,
        float(scale),
    )

    ct_stream = cp.cuda.get_current_stream()
    ct.launch(ct_stream, grid, kernel, kernel_args)
    return out_gpu


def launch(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
):
    """Copy inputs to the GPU and launch fused causal attention."""
    cp, _ = _import_runtime()

    tile = _validate_cutile_config(q, k, v, tile_q, tile_k)
    q_gpu = cp.asarray(q)
    k_gpu = cp.asarray(k)
    v_gpu = cp.asarray(v)
    return launch_device(
        q_gpu,
        k_gpu,
        v_gpu,
        tile_q=tile.query,
        tile_k=tile.key,
    )

def forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
) -> np.ndarray:
    """Compute fused causal attention and return a NumPy FP16 output."""
    cp = _import_runtime()[0]
    out_gpu = launch(q, k, v, tile_q=tile_q, tile_k=tile_k)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)
