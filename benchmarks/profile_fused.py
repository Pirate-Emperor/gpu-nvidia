#!/usr/bin/env python3
"""Launch one warmed cuTile attention kernel for Nsight Compute profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cases import MANDATORY_CASES
from cutile_attention import launch_device


def make_inputs(case, cp):
    rng = np.random.default_rng(case.seed)
    q = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(
        np.float16
    )
    k = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(
        np.float16
    )
    v = rng.standard_normal(
        (case.B, case.H, case.N, case.Dv), dtype=np.float32
    ).astype(np.float16)
    return cp.asarray(q), cp.asarray(k), cp.asarray(v)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="medium_d128", choices=[c.name for c in MANDATORY_CASES])
    parser.add_argument("--tile-q", type=int, default=64)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if args.warmup < 1:
        parser.error("--warmup must be >= 1")

    case = next(case for case in MANDATORY_CASES if case.name == args.case)
    if case.N % args.tile_q or case.N % args.tile_k:
        parser.error(
            f"{case.name}: N={case.N} must be divisible by "
            f"tile_q={args.tile_q} and tile_k={args.tile_k}"
        )

    try:
        import cupy as cp
        import cuda.tile  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: cuTile/CuPy is required: {exc}", file=sys.stderr)
        return 2

    q, k, v = make_inputs(case, cp)
    stream = cp.cuda.get_current_stream()

    # The first call is intentionally outside the profiled steady-state region.
    launch_device(q, k, v, tile_q=args.tile_q, tile_k=args.tile_k)
    stream.synchronize()

    for _ in range(args.warmup):
        launch_device(q, k, v, tile_q=args.tile_q, tile_k=args.tile_k)
    stream.synchronize()

    print(
        f"Ready for profiling: case={case.name} "
        f"B={case.B} H={case.H} N={case.N} D={case.D} "
        f"tile={args.tile_q}x{args.tile_k}"
    )

    # Exactly one steady-state fused launch for Nsight Compute.
    launch_device(q, k, v, tile_q=args.tile_q, tile_k=args.tile_k)
    stream.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
