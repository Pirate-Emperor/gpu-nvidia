#!/usr/bin/env python3
"""Verify the saved CLI output against the independent NumPy attention oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from attention_io import load_attention_npz
from reference import causal_attention_fp32, passes_mixed_tolerance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    q, k, v = load_attention_npz(args.input)
    actual = np.load(args.output)
    expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)

    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: actual={actual.shape}, expected={expected.shape}")
    if actual.dtype != np.float16:
        raise AssertionError(f"output dtype must be float16, got {actual.dtype}")
    if not np.isfinite(actual).all():
        raise AssertionError("CLI output contains NaN or infinity")
    if not passes_mixed_tolerance(actual, expected, abs_scale=2e-2, rel_scale=2e-2):
        delta = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
        raise AssertionError(f"CLI output mismatches reference; max_abs_error={float(delta.max())}")

    print("CLI_REFERENCE_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
