"""NumPy reference for tiled online-softmax attention."""

from __future__ import annotations

import math

import numpy as np

from reference import validate_attention_inputs


def causal_attention_online_fp32(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_q: int,
    tile_k: int,
) -> np.ndarray:
    """Compute causal attention with the same streaming recurrence as the kernel.

    All arithmetic is FP32 after the FP16 inputs are converted. Q/K/V tiles are
    processed one at a time and the full attention score/probability matrix is
    never constructed.
    """
    validate_attention_inputs(q, k, v)

    if tile_q <= 0 or tile_k <= 0:
        raise ValueError("tile sizes must be positive")
    if q.shape[-2] % tile_q != 0 or q.shape[-2] % tile_k != 0:
        raise ValueError("sequence length must be divisible by both tile sizes")

    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    v32 = v.astype(np.float32)

    bsz, heads, seq_len, d = q.shape
    dv = v.shape[-1]
    scale = np.float32(1.0 / math.sqrt(d))

    output = np.empty((bsz, heads, seq_len, dv), dtype=np.float32)

    for b in range(bsz):
        for h in range(heads):
            for q0 in range(0, seq_len, tile_q):
                qtile = q32[b, h, q0 : q0 + tile_q]
                running_max = np.full((tile_q, 1), -np.inf, dtype=np.float32)
                running_norm = np.zeros((tile_q, 1), dtype=np.float32)
                running_out = np.zeros((tile_q, dv), dtype=np.float32)

                q_pos = np.arange(q0, q0 + tile_q, dtype=np.int32)[:, None]

                for k0 in range(0, seq_len, tile_k):
                    ktile = k32[b, h, k0 : k0 + tile_k]
                    vtile = v32[b, h, k0 : k0 + tile_k]

                    scores = (qtile @ ktile.T) * scale
                    k_pos = np.arange(k0, k0 + tile_k, dtype=np.int32)[None, :]
                    scores = np.where(
                        k_pos > q_pos,
                        np.float32(-np.inf),
                        scores,
                    )

                    tile_max = np.max(scores, axis=1, keepdims=True)
                    new_max = np.maximum(running_max, tile_max)

                    # A whole K-tile can be future for an entire Q-tile.
                    # In that case both maxima are -inf; evaluating
                    # exp(-inf - (-inf)) would create NaNs. Such a tile must
                    # leave the running state unchanged.
                    tile_valid = np.isfinite(tile_max)
                    old_valid = np.isfinite(running_max)

                    rescale_arg = np.where(
                        old_valid,
                        running_max - new_max,
                        np.float32(-np.inf),
                    )
                    rescale = np.where(
                        tile_valid,
                        np.exp(rescale_arg, dtype=np.float32),
                        np.float32(1.0),
                    )

                    finite_scores = np.isfinite(scores)
                    weight_arg = np.where(
                        finite_scores,
                        scores - new_max,
                        np.float32(0.0),
                    )
                    weights = np.where(
                        finite_scores,
                        np.exp(weight_arg, dtype=np.float32),
                        np.float32(0.0),
                    )

                    running_norm = (
                        rescale * running_norm
                        + np.sum(weights, axis=1, keepdims=True, dtype=np.float32)
                    )
                    running_out = (
                        rescale * running_out
                        + weights @ vtile
                    )
                    running_max = new_max

                output[b, h, q0 : q0 + tile_q] = running_out / running_norm

    if not np.isfinite(output).all():
        raise FloatingPointError("online reference produced NaN or infinity")

    return output
