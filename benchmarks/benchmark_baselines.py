#!/usr/bin/env python3
"""Benchmark the exploratory GPU baselines on manageable sequence lengths."""

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
from naive_cuda_attention import launch_device as naive_launch

def make_inputs(case, cp):
    rng = np.random.default_rng(case.seed)
    q = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32).astype(np.float16)
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
    parser.add_argument("--case", default="short_d32", choices=[case.name for case in MANDATORY_CASES])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("baseline_results.json"))
    args = parser.parse_args()

    if args.iterations < 30:
        parser.error("--iterations must be >= 30")

    import cupy as cp

    case = next(case for case in MANDATORY_CASES if case.name == args.case)
    if case.N > 512:
        parser.error("The scalar CUDA baseline is intentionally slow; use N <= 512")

    q, k, v = make_inputs(case, cp)
    q32, k32, v32 = q.astype(cp.float32), k.astype(cp.float32), v.astype(cp.float32)

    naive_ms, naive_p90 = timed(lambda: naive_launch(q, k, v), cp, args.warmup, args.iterations)
    unfused_ms, unfused_p90 = timed(lambda: unfused_launch(q32, k32, v32), cp, args.warmup, args.iterations)

    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(name, bytes):
        name = name.decode(errors="replace")

    payload = {
        "case": case.name,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "gpu": name,
        "naive_cuda_median_ms": naive_ms,
        "naive_cuda_p90_ms": naive_p90,
        "unfused_cupy_median_ms": unfused_ms,
        "unfused_cupy_p90_ms": unfused_p90,
        "naive_over_unfused_speedup": unfused_ms / naive_ms,
    }

    args.output.write_text(json.dumps(payload, indent=2) + "\\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload.keys())
        writer.writeheader()
        writer.writerow(payload)

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
