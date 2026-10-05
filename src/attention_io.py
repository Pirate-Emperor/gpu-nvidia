"""Input/output helpers for the course-defined NPZ/NPY interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reference import AttentionValidationError, validate_attention_inputs


def load_attention_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Q/K/V from an .npz file and validate the common tensor contract."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")

    with np.load(input_path, allow_pickle=False) as data:
        required = {"Q", "K", "V"}
        keys = set(data.files)
        missing = required - keys
        extra = keys - required
        if missing:
            raise AttentionValidationError(
                f"input is missing required arrays: {sorted(missing)}"
            )
        if extra:
            raise AttentionValidationError(
                f"input contains unsupported arrays: {sorted(extra)}"
            )

        q = np.ascontiguousarray(data["Q"])
        k = np.ascontiguousarray(data["K"])
        v = np.ascontiguousarray(data["V"])

    validate_attention_inputs(q, k, v)
    return q, k, v


def save_attention_npy(path: str | Path, output: np.ndarray) -> None:
    """Write one .npy array, rejecting non-finite outputs."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(output, np.ndarray):
        raise TypeError("output must be a NumPy array")
    if not np.all(np.isfinite(output)):
        raise ValueError("output contains NaN or infinity")

    np.save(output_path, output)


def make_npz(
    path: str | Path,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> None:
    """Create a course-format NPZ after validating Q/K/V."""
    q = np.ascontiguousarray(q)
    k = np.ascontiguousarray(k)
    v = np.ascontiguousarray(v)
    validate_attention_inputs(q, k, v)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, Q=q, K=k, V=v)
