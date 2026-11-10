#!/usr/bin/env python3
"""Prepare one steady-state current shared-tile CUDA attention launch for Nsight Compute."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from cases import MANDATORY_CASES
from shared_tile_cuda_attention import launch_device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=[c.name for c in MANDATORY_CASES], required=True)
    ap.add_argument("--tile-k", type=int, choices=(16, 32, 64), required=True)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    if args.warmup < 1:
        ap.error("--warmup must be >= 1")

    import cupy as cp

    case = next(c for c in MANDATORY_CASES if c.name == args.case)
    if case.N % args.tile_k:
        ap.error(f"{case.name}: N={case.N} must be divisible by tile_k={args.tile_k}")

    rng = np.random.default_rng(case.seed)
    q = cp.asarray(rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16))
    k = cp.asarray(rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16))
    v = cp.asarray(rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32).astype(np.float16))
    stream = cp.cuda.get_current_stream()

    # Trigger JIT outside the profiled steady-state region.
    launch_device(q, k, v, tile_k=args.tile_k)
    stream.synchronize()

    for _ in range(args.warmup):
        launch_device(q, k, v, tile_k=args.tile_k)
    stream.synchronize()

    print(
        f"Ready: case={case.name} B={case.B} H={case.H} "
        f"N={case.N} D={case.D} tile_k={args.tile_k}"
    )

    # Exactly one warmed fused launch for Nsight Compute.
    launch_device(q, k, v, tile_k=args.tile_k)
    stream.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
