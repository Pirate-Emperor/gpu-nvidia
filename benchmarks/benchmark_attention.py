#!/usr/bin/env python3
"""Benchmark fused cuTile attention against the deliberately unfused GPU baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from baseline import launch_device as baseline_launch_device
from cases import MANDATORY_CASES
from cutile_attention import launch_device as cutile_launch_device


def make_inputs(case, cp):
    rng = np.random.default_rng(case.seed)
    q = (
        rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32)
        .astype(np.float16)
    )
    k = (
        rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32)
        .astype(np.float16)
    )
    v = (
        rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32)
        .astype(np.float16)
    )
    return cp.asarray(q), cp.asarray(k), cp.asarray(v)


def timed_gpu(fn, cp, warmup: int, iterations: int) -> tuple[float, float]:
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

    samples = np.asarray(samples, dtype=np.float64)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def cold_start_ms(fn, cp) -> float:
    stream = cp.cuda.get_current_stream()
    start = time.perf_counter()
    fn()
    stream.synchronize()
    return (time.perf_counter() - start) * 1000.0


def parse_tile(value: str) -> tuple[int, int]:
    try:
        q, k = (int(part) for part in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid tile '{value}', expected TILE_QxTILE_K such as 64x64"
        ) from exc
    if q <= 0 or k <= 0:
        raise argparse.ArgumentTypeError("tile dimensions must be positive")
    return q, k


def run_case(case, cp, tiles, warmup, iterations):
    q, k, v = make_inputs(case, cp)
    q32 = q.astype(cp.float32)
    k32 = k.astype(cp.float32)
    v32 = v.astype(cp.float32)
    output_bytes = case.B * case.H * case.N * case.Dv * np.dtype(np.float16).itemsize
    rows = []

    for tile_q, tile_k in tiles:
        if case.N % tile_q or case.N % tile_k:
            continue

        fused_call = lambda: cutile_launch_device(
            q, k, v, tile_q=tile_q, tile_k=tile_k
        )
        baseline_call = lambda: baseline_launch_device(q32, k32, v32)

        jit_ms = cold_start_ms(fused_call, cp)
        fused_ms, fused_p90 = timed_gpu(
            fused_call, cp, warmup=warmup, iterations=iterations
        )
        baseline_ms, baseline_p90 = timed_gpu(
            baseline_call, cp, warmup=warmup, iterations=iterations
        )

        rows.append(
            {
                "case": case.name,
                "B": case.B,
                "H": case.H,
                "N": case.N,
                "D": case.D,
                "Dv": case.Dv,
                "tile_q": tile_q,
                "tile_k": tile_k,
                "jit_ms": jit_ms,
                "fused_median_ms": fused_ms,
                "fused_p90_ms": fused_p90,
                "baseline_median_ms": baseline_ms,
                "baseline_p90_ms": baseline_p90,
                "speedup_vs_unfused": baseline_ms / fused_ms,
                "fused_output_GiB_s": output_bytes / fused_ms / 2**30 * 1000.0,
                "baseline_output_GiB_s": output_bytes / baseline_ms / 2**30 * 1000.0,
                "iterations": iterations,
                "warmup": warmup,
            }
        )

    del q, k, v, q32, k32, v32
    cp.get_default_memory_pool().free_all_blocks()
    return rows


def environment_metadata(cp, gpu_name):
    metadata = {
        "gpu": gpu_name,
        "cupy": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cutile": importlib.metadata.version("cuda-tile"),
        "compilation_mode": "JIT excluded from steady-state timing; cold-start reported separately",
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.current.sm,power.limit,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        metadata["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["nvidia_smi"] = None
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="JSON output path; a sibling CSV is also written",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--tiles",
        nargs="+",
        type=parse_tile,
        default=[(32, 64), (64, 64)],
        metavar="QxK",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in MANDATORY_CASES],
        help="benchmark only the selected case; repeat the option for multiple cases",
    )
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 30:
        parser.error("--warmup must be >= 1 and --iterations must be >= 30")

    try:
        import cupy as cp
    except ImportError as exc:
        print(f"ERROR: CuPy is required for benchmarking: {exc}", file=sys.stderr)
        return 2

    try:
        import cuda.tile  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: cuTile is required for benchmarking: {exc}", file=sys.stderr)
        return 2

    device = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = device.get("name", b"")
    gpu_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)

    selected_cases = MANDATORY_CASES
    if args.case:
        selected_names = set(args.case)
        selected_cases = tuple(case for case in MANDATORY_CASES if case.name in selected_names)

    all_rows = []
    for case in selected_cases:
        all_rows.extend(
            run_case(
                case,
                cp,
                args.tiles,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )

    if not all_rows:
        print("ERROR: no benchmark case is divisible by the requested tile configurations")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": environment_metadata(cp, gpu_name),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tiles": [{"tile_q": q, "tile_k": k} for q, k in args.tiles],
        "results": all_rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"GPU: {gpu_name}")
    print(f"Results: {args.output}")
    print(f"CSV: {csv_path}")
    for row in all_rows:
        print(
            f"{row['case']:24s} tile={row['tile_q']}x{row['tile_k']} "
            f"fused={row['fused_median_ms']:.3f} ms "
            f"baseline={row['baseline_median_ms']:.3f} ms "
            f"speedup={row['speedup_vs_unfused']:.2f}x "
            f"jit={row['jit_ms']:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
