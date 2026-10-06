#!/usr/bin/env python3
"""Benchmark the non-cuTile GPU baseline ladder on manageable configurations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from baseline import launch_device as unfused_launch
from cases import MANDATORY_CASES
from custom_cuda_attention import launch_device as custom_block_launch
from naive_cuda_attention import launch_device as naive_launch
from shared_tile_cuda_attention import launch_device as shared_tile_launch
from warp_cuda_attention import launch_device as custom_warp_launch


def make_inputs(case, cp):
    rng = np.random.default_rng(case.seed)
    q = rng.standard_normal(
        (case.B, case.H, case.N, case.D), dtype=np.float32
    ).astype(np.float16)
    k = rng.standard_normal(
        (case.B, case.H, case.N, case.D), dtype=np.float32
    ).astype(np.float16)
    v = rng.standard_normal(
        (case.B, case.H, case.N, case.Dv), dtype=np.float32
    ).astype(np.float16)
    return cp.asarray(q), cp.asarray(k), cp.asarray(v)


def timed(fn, cp, warmup, iterations):
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        fn()
    stream.synchronize()

    samples = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)))

    values = np.asarray(samples, dtype=np.float64)
    return float(np.median(values)), float(np.percentile(values, 90))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="short_d32",
        choices=[case.name for case in MANDATORY_CASES],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("baseline_results.json"))
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="skip the intentionally slow scalar CUDA baseline",
    )
    args = parser.parse_args()

    if args.iterations < 30:
        parser.error("--iterations must be >= 30")

    import cupy as cp

    case = next(case for case in MANDATORY_CASES if case.name == args.case)
    if case.N > 512 and not args.skip_naive:
        parser.error(
            "The scalar CUDA baseline is intentionally slow for N>512; "
            "rerun with --skip-naive for the fused CUDA variants."
        )

    q, k, v = make_inputs(case, cp)
    q32, k32, v32 = q.astype(cp.float32), k.astype(cp.float32), v.astype(cp.float32)

    payload = {
        "case": case.name,
        "B": case.B,
        "H": case.H,
        "N": case.N,
        "D": case.D,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }

    block_ms, block_p90 = timed(
        lambda: custom_block_launch(q, k, v), cp, args.warmup, args.iterations
    )
    warp_ms, warp_p90 = timed(
        lambda: custom_warp_launch(q, k, v), cp, args.warmup, args.iterations
    )
    tiled_ms, tiled_p90 = timed(
        lambda: shared_tile_launch(q, k, v, tile_k=64),
        cp,
        args.warmup,
        args.iterations,
    )
    unfused_ms, unfused_p90 = timed(
        lambda: unfused_launch(q32, k32, v32), cp, args.warmup, args.iterations
    )

    payload.update(
        {
            "custom_block_median_ms": block_ms,
            "custom_block_p90_ms": block_p90,
            "custom_warp_median_ms": warp_ms,
            "custom_warp_p90_ms": warp_p90,
            "shared_tile64_median_ms": tiled_ms,
            "shared_tile64_p90_ms": tiled_p90,
            "unfused_cupy_median_ms": unfused_ms,
            "unfused_cupy_p90_ms": unfused_p90,
            "custom_block_over_unfused_speedup": unfused_ms / block_ms,
            "custom_warp_over_unfused_speedup": unfused_ms / warp_ms,
            "shared_tile64_over_unfused_speedup": unfused_ms / tiled_ms,
            "custom_warp_over_custom_block_speedup": block_ms / warp_ms,
            "shared_tile64_over_custom_block_speedup": block_ms / tiled_ms,

        }
    )

    if not args.skip_naive:
        naive_ms, naive_p90 = timed(
            lambda: naive_launch(q, k, v), cp, args.warmup, args.iterations
        )
        payload.update(
            {
                "naive_cuda_median_ms": naive_ms,
                "naive_cuda_p90_ms": naive_p90,
                "naive_over_unfused_speedup": unfused_ms / naive_ms,
                "custom_block_over_naive_speedup": naive_ms / block_ms,
                "custom_warp_over_naive_speedup": naive_ms / warp_ms,
            }
        )

    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    payload["gpu"] = name

    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload.keys())
        writer.writeheader()
        writer.writerow(payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
