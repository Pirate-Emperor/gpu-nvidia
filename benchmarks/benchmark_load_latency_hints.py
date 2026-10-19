#!/usr/bin/env python3
"""Test cuTile load-latency hints against compiler-inferred default latency.

This is a narrow profiler-guided experiment for SM89. It compares identical
inputs, first verifies the NumPy mixed-tolerance gate, then runs three
order-rotated GPU-event sessions. The compiler-default (hint 0) remains the
decision baseline; a hint is only recommended when its median lead clears a
noise-aware threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cases import MANDATORY_CASES
from cutile_attention import launch_device
from reference import causal_attention_fp32, passes_mixed_tolerance

CONFIGS = (
    ("medium_d128", 64, 64),
    ("longer_batched_d128", 64, 128),
)
LATENCY_HINTS = (0, 2, 4, 6, 8, 10)
ABS_TOL = 2e-2
REL_TOL = 2e-2


def make_inputs(case):
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
    return q, k, v


def event_measure(fn, cp, warmup, iterations, repeats):
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        fn()
    stream.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record(stream)
        for _ in range(repeats):
            fn()
        end.record(stream)
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)) / repeats)
    a = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_sample_ms": float(a.min()),
        "max_sample_ms": float(a.max()),
        "samples_ms": [float(x) for x in a],
    }


def hint_recommendation(rows, minimum_gain_pct=1.0):
    """Rows map hint value -> per-session median timing statistics."""
    baseline_rows = rows.get(0, [])
    if len(baseline_rows) < 3:
        return {"latency_hint": 0, "credible": False, "status": "missing three baseline sessions"}
    baseline_medians = [float(x["median_ms"]) for x in baseline_rows]
    baseline_median = float(statistics.median(baseline_medians))
    baseline_range_pct = 100.0 * (max(baseline_medians)-min(baseline_medians))/baseline_median
    candidates = []
    for hint, items in rows.items():
        if hint == 0 or len(items) < 3:
            continue
        medians = [float(x["median_ms"]) for x in items]
        median = float(statistics.median(medians))
        spread_pct = 100.0 * (max(medians)-min(medians))/median
        gain = 100.0 * (baseline_median - median) / baseline_median
        needed = max(minimum_gain_pct, 0.5 * (baseline_range_pct + spread_pct))
        candidates.append((gain, needed, hint, median, spread_pct))
    if not candidates:
        return {"latency_hint": 0, "credible": False, "status": "no complete candidate sessions"}
    gain, needed, hint, median, spread_pct = max(candidates, key=lambda x: x[0])
    if gain > needed:
        return {
            "latency_hint": hint, "credible": True,
            "status": "hint clears noise-aware gain threshold",
            "gain_pct_vs_default": gain, "required_gain_pct": needed,
            "median_ms": median, "session_range_pct": spread_pct,
            "default_median_ms": baseline_median,
            "default_session_range_pct": baseline_range_pct,
        }
    return {
        "latency_hint": 0, "credible": False,
        "status": "no hint beats the compiler-inferred default beyond observed spread",
        "best_candidate_hint": hint, "best_candidate_gain_pct": gain,
        "required_gain_pct": needed, "default_median_ms": baseline_median,
        "default_session_range_pct": baseline_range_pct,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/cutile-load-latency-hints.json"))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--repeats-per-sample", type=int, default=5)
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--minimum-gain-pct", type=float, default=1.0)
    args = p.parse_args()
    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1 or args.sessions < 3:
        p.error("require warmup >= 1, iterations >= 30, repeats >= 1, and sessions >= 3")

    try:
        import cupy as cp
        import cuda.tile  # noqa: F401
    except Exception as exc:
        print(f"ERROR: CuPy/cuTile import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    case_map = {c.name: c for c in MANDATORY_CASES}
    all_trials = []
    for case_name, tq, tk in CONFIGS:
        case = case_map[case_name]
        q_np, k_np, v_np = make_inputs(case)
        expected = causal_attention_fp32(q_np, k_np, v_np, output_dtype=np.float16)
        q, k, v = cp.asarray(q_np), cp.asarray(k_np), cp.asarray(v_np)

        # Accuracy is checked for every specialization before any timing.
        for hint in LATENCY_HINTS:
            output = launch_device(q, k, v, tile_q=tq, tile_k=tk, load_latency=hint)
            cp.cuda.get_current_stream().synchronize()
            actual = cp.asnumpy(output)
            if actual.dtype != np.float16 or not np.isfinite(actual).all():
                raise AssertionError(f"{case_name}/{tq}x{tk}/latency={hint}: dtype or finiteness failed")
            if not passes_mixed_tolerance(actual, expected, abs_scale=ABS_TOL, rel_scale=REL_TOL):
                raise AssertionError(f"{case_name}/{tq}x{tk}/latency={hint}: independent NumPy tolerance failed")

        session_orders = []
        for session in range(args.sessions):
            order = list(LATENCY_HINTS)
            shift = session % len(order)
            order = order[shift:] + order[:shift]
            if session % 2:
                order.reverse()
            session_orders.append(order)
            for hint in order:
                def fn(hint=hint):
                    return launch_device(q, k, v, tile_q=tq, tile_k=tk, load_latency=hint)

                # Capture a first-call number only for the first appearance of
                # each config across the whole process; do not confuse it with
                # steady-state CUDA-event samples.
                first_call = None
                if not any(
                    x["case"] == case_name and x["tile_q"] == tq and
                    x["tile_k"] == tk and x["latency_hint"] == hint
                    for x in all_trials
                ):
                    start = time.perf_counter()
                    fn()
                    cp.cuda.get_current_stream().synchronize()
                    first_call = (time.perf_counter() - start) * 1000.0

                timed = event_measure(fn, cp, args.warmup, args.iterations, args.repeats_per_sample)
                all_trials.append({
                    "case": case_name, "B": case.B, "H": case.H, "N": case.N,
                    "D": case.D, "tile_q": tq, "tile_k": tk,
                    "latency_hint": hint, "session": session + 1,
                    "hint_order": order, "correctness_pass": True,
                    "first_call_ms": first_call,
                    **timed,
                    "warmup": args.warmup, "iterations": args.iterations,
                    "repeats_per_sample": args.repeats_per_sample,
                })
        del q, k, v
        cp.get_default_memory_pool().free_all_blocks()

    grouped = defaultdict(dict)
    for row in all_trials:
        grouped[(row["case"], row["tile_q"], row["tile_k"])].setdefault(
            row["latency_hint"], []
        ).append(row)
    recommendations = []
    for key, by_hint in grouped.items():
        rec = hint_recommendation(by_hint, minimum_gain_pct=args.minimum_gain_pct)
        recommendations.append({
            "case": key[0], "tile_q": key[1], "tile_k": key[2], **rec
        })

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props.get("name", b"")
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    payload = {
        "title": "cuTile DRAM load-latency-hint experiment",
        "environment": {
            "gpu": str(gpu_name),
            "compute_capability": "SM89 expected for this experiment",
            "cupy": cp.__version__,
            "cuda_tile": __import__("importlib.metadata", fromlist=["version"]).version("cuda-tile"),
            "warmup": args.warmup, "iterations": args.iterations,
            "repeats_per_sample": args.repeats_per_sample,
            "sessions": args.sessions,
            "hints": list(LATENCY_HINTS),
            "baseline_hint": 0,
        },
        "method": "All hint variants pass independent FP32 NumPy mixed tolerance before timing. Three order-rotated sessions use 10 warmups/30 CUDA-event samples/5 launches per sample by default. A hint is recommended only if its median gain exceeds max(minimum_gain_pct, half the combined baseline/candidate three-session range).",
        "recommendations": recommendations,
        "raw_trials": all_trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["case","tile_q","tile_k","latency_hint","session","median_ms","p90_ms","min_sample_ms","max_sample_ms","first_call_ms","correctness_pass","warmup","iterations","repeats_per_sample"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in all_trials:
            w.writerow({key: row.get(key) for key in fields})
    print(json.dumps(recommendations, indent=2))
    print(f"LATENCY_HINT_JSON={args.output}")
    print(f"LATENCY_HINT_CSV={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
