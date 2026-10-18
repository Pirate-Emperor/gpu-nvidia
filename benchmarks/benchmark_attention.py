#!/usr/bin/env python3
"""Benchmark fused cuTile attention against the deliberately unfused GPU baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
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


def timed_gpu(
    fn,
    cp,
    warmup: int,
    iterations: int,
    repeats_per_sample: int,
) -> tuple[float, float]:
    """Measure 30+ event samples, each averaging several identical invocations.

    A short kernel can be near the timer resolution and one event sample is
    sensitive to scheduling noise. Grouping 5 calls per event makes each
    reported sample longer while preserving per-invocation latency.
    """
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        fn()
    stream.synchronize()

    samples = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        for _repeat in range(repeats_per_sample):
            fn()
        end.record(stream)
        end.synchronize()
        elapsed = float(cp.cuda.get_elapsed_time(start, end))
        samples.append(elapsed / repeats_per_sample)

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


def run_case(
    case, cp, tiles, warmup, iterations, repeats_per_sample, seen_specializations
):
    q, k, v = make_inputs(case, cp)
    q32 = q.astype(cp.float32)
    k32 = k.astype(cp.float32)
    v32 = v.astype(cp.float32)
    output_bytes = case.B * case.H * case.N * case.Dv * np.dtype(np.float16).itemsize
    rows = []
    baseline_call = lambda: baseline_launch_device(q32, k32, v32)

    # Bracket the tile sweep with unfused-baseline measurements. This avoids
    # timing the same baseline separately after every tile and lets us detect
    # drift during the session rather than confounding it with tile order.
    baseline_before_ms, baseline_before_p90 = timed_gpu(
        baseline_call, cp, warmup=warmup, iterations=iterations,
        repeats_per_sample=repeats_per_sample,
    )

    for tile_q, tile_k in tiles:
        if case.N % tile_q or case.N % tile_k:
            continue

        fused_call = lambda: cutile_launch_device(
            q, k, v, tile_q=tile_q, tile_k=tile_k
        )
        specialization = (tile_q, tile_k, case.D, case.Dv)
        if specialization in seen_specializations:
            # Each benchmark process reports first-call timing only once per
            # specialization. This can still be a compiler-cache hit.
            jit_ms = None
        else:
            jit_ms = cold_start_ms(fused_call, cp)
            seen_specializations.add(specialization)
        fused_ms, fused_p90 = timed_gpu(
            fused_call, cp, warmup=warmup, iterations=iterations,
            repeats_per_sample=repeats_per_sample,
        )

        rows.append(
            {
                "case": case.name,
                "B": case.B,
                "H": case.H,
                "N": case.N,
                "D": case.D,
                "Dv": case.Dv,
                "seed": case.seed,
                "tile_q": tile_q,
                "tile_k": tile_k,
                "jit_ms": jit_ms,
                "fused_median_ms": fused_ms,
                "fused_p90_ms": fused_p90,
                "baseline_before_median_ms": baseline_before_ms,
                "baseline_before_p90_ms": baseline_before_p90,
                "baseline_median_ms": None,
                "baseline_p90_ms": None,
                "speedup_vs_unfused": None,
                "fused_output_GiB_s": output_bytes / fused_ms / 2**30 * 1000.0,
                "baseline_output_GiB_s": None,
                "iterations": iterations,
                "warmup": warmup,
                "repeats_per_sample": repeats_per_sample,
                "measured_launches": iterations * repeats_per_sample,
            }
        )

    baseline_after_ms, baseline_after_p90 = timed_gpu(
        baseline_call, cp, warmup=warmup, iterations=iterations,
        repeats_per_sample=repeats_per_sample,
    )
    baseline_ms = (baseline_before_ms + baseline_after_ms) / 2.0
    baseline_p90 = (baseline_before_p90 + baseline_after_p90) / 2.0
    baseline_bracket_drift_pct = (
        100.0 * abs(baseline_after_ms - baseline_before_ms) / baseline_ms
        if baseline_ms > 0 else 0.0
    )
    for row in rows:
        row["baseline_after_median_ms"] = baseline_after_ms
        row["baseline_after_p90_ms"] = baseline_after_p90
        row["baseline_bracket_drift_pct"] = baseline_bracket_drift_pct
        row["baseline_median_ms"] = baseline_ms
        row["baseline_p90_ms"] = baseline_p90
        row["speedup_vs_unfused"] = baseline_ms / row["fused_median_ms"]
        row["baseline_output_GiB_s"] = output_bytes / baseline_ms / 2**30 * 1000.0

    del q, k, v, q32, k32, v32
    cp.get_default_memory_pool().free_all_blocks()
    return rows


def environment_metadata(cp, gpu_name):
    metadata = {
        "gpu": gpu_name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cupy": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cutile": importlib.metadata.version("cuda-tile"),
        "compilation_mode": "First-call wall time is reported once per (tile_q, tile_k, D, Dv) specialization; it may include a compiler-cache hit. Steady-state CUDA-event timing excludes that first call.",
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
    try:
        metadata["git_head_observed"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["git_head_observed"] = None
    metadata["git_commit"] = os.environ.get("GITHUB_SHA") or metadata["git_head_observed"]
    metadata["git_commit_source"] = (
        "GITHUB_SHA" if os.environ.get("GITHUB_SHA") else "git rev-parse HEAD"
    )
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
    parser.add_argument("--iterations", type=int, default=30)
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
    parser.add_argument("--repeats-per-sample", type=int, default=5, help="GPU launches inside each CUDA-event sample")
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1:
        parser.error("--warmup must be >= 1, --iterations >= 30, and --repeats-per-sample >= 1")

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
    seen_specializations = set()
    for case in selected_cases:
        all_rows.extend(
            run_case(
                case,
                cp,
                args.tiles,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats_per_sample=args.repeats_per_sample,
                seen_specializations=seen_specializations,
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
        "repeats_per_sample": args.repeats_per_sample,
        "timed_launches_per_configuration": args.iterations * args.repeats_per_sample,
        "baseline_measurement": "baseline median/P90 is the average of measurements bracketed around the tile sweep",
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
        first_call = (
            f"{row['jit_ms']:.1f} ms"
            if row["jit_ms"] is not None
            else "already compiled"
        )
        print(
            f"{row['case']:24s} tile={row['tile_q']}x{row['tile_k']} "
            f"fused={row['fused_median_ms']:.3f} ms "
            f"baseline={row['baseline_median_ms']:.3f} ms "
            f"speedup={row['speedup_vs_unfused']:.2f}x "
            f"first_call={first_call}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
