"""Mandatory launch-contract validation shared by the GPU backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


@dataclass(frozen=True)
class TileConfig:
    query: int
    key: int

    def validate(self) -> None:
        if self.query <= 0 or self.key <= 0:
            raise AttentionValidationError("tile dimensions must be positive")


def validate_launch_contract(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    tile: TileConfig,
) -> None:
    """Validate all shape constraints known before a cuTile launch.

    Kernel-specific resource/feature support is deliberately left to the cuTile
    backend. This validator only enforces mathematical and project-contract
    invariants that are backend-independent.
    """
    tile.validate()
    _, _, nq, nk, _ = validate_attention_inputs(q, k, v)

    if nq % tile.query != 0:
        raise AttentionValidationError(
            f"query sequence length {nq} is not divisible by query tile {tile.query}"
        )
    if nk % tile.key != 0:
        raise AttentionValidationError(
            f"key sequence length {nk} is not divisible by key tile {tile.key}"
        )
