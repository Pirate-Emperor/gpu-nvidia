#!/usr/bin/env python3
"""Try legal tile pairs. Recommend a tile only when repeated timings clear the noise threshold."""
from __future__ import annotations

import argparse
import csv
import json
import math
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

CANDIDATE_TILES = ((32, 64), (64, 64), (64, 128), (128, 64), (128, 128))
DEFAULT_TILE = (64, 64)
ABS_TOL = 2e-2
REL_TOL = 2e-2


def recommend_tile(tile_summaries, default_tile=DEFAULT_TILE, min_gain_pct=1.0):
    """Pure helper tested by CPU CI; summaries map tile strings to timing rows."""
    if not tile_summaries:
        return {
            "tile_q": default_tile[0], "tile_k": default_tile[1],
            "credible": False, "status": "no legal tile measurements",
            "gain_pct_vs_default": None,
        }

    def key(tile):
        return f"{tile[0]}x{tile[1]}"

    default_key = key(default_tile)
    default = tile_summaries.get(default_key)
    best_key, best = min(
        tile_summaries.items(), key=lambda item: float(item[1]["median_ms"])
    )
    best_q, best_k = (int(x) for x in best_key.split("x"))
    if default is None:
        return {
            "tile_q": best_q, "tile_k": best_k, "credible": False,
            "status": "default tile was not legal/measured; inspect the full sweep",
            "gain_pct_vs_default": None,
        }
    default_ms = float(default["median_ms"])
    best_ms = float(best["median_ms"])
    gain_pct = 100.0 * (default_ms - best_ms) / default_ms if default_ms else 0.0
    noise_floor = max(
        float(min_gain_pct),
        0.5 * (
            float(default.get("range_pct", 0.0)) +
            float(best.get("range_pct", 0.0))
        ),
    )
    credible = best_ms < default_ms and gain_pct > noise_floor
    if best_key == default_key:
        credible = True
        status = "default tile remains fastest by median"
    elif credible:
        status = "candidate improvement exceeds configured spread/margin rule"
    else:
        best_q, best_k = default_tile
        status = "inconclusive within repeated-session spread; retain default"
    return {
        "tile_q": best_q, "tile_k": best_k,
        "credible": bool(credible), "status": status,
        "gain_pct_vs_default": float(gain_pct),
        "required_gain_pct": float(noise_floor),
    }


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


def timed_cupy(fn, cp, warmup, iterations, repeats_per_sample):
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        fn()
    stream.synchronize()
    samples = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record(stream)
        for _ in range(repeats_per_sample):
            fn()
        end.record(stream)
        end.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, end)) / repeats_per_sample)
    a = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_sample_ms": float(a.min()),
        "max_sample_ms": float(a.max()),
        "samples_ms": [float(x) for x in a],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cutile-autotune.json"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    parser.add_argument("--min-gain-pct", type=float, default=1.0)
    parser.add_argument("--sessions", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1:
        parser.error("warmup >= 1, iterations >= 30, repeats-per-sample >= 1 required")
    if args.sessions < 3 or args.min_gain_pct < 0:
        parser.error("sessions must be >= 3 and min-gain-pct must be nonnegative")

    try:
        import cupy as cp
        import cuda.tile  # noqa: F401
    except Exception as exc:
        print(f"ERROR: CuPy/cuTile imports required for autotuning: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    cases = MANDATORY_CASES
    all_trials = []
    cold_calls = set()

    for case in cases:
        q_np, k_np, v_np = make_inputs(case)
        expected = causal_attention_fp32(q_np, k_np, v_np, output_dtype=np.float16)
        q = cp.asarray(q_np)
        k = cp.asarray(k_np)
        v = cp.asarray(v_np)
        legal = [tile for tile in CANDIDATE_TILES if case.N % tile[0] == 0 and case.N % tile[1] == 0]
        if DEFAULT_TILE not in legal:
            # This frozen workload matrix includes 64x64 for every required case.
            raise RuntimeError(f"default tile 64x64 is unexpectedly illegal for {case.name}")

        for session in range(1, args.sessions + 1):
            order = list(legal)
            # Deterministic rotations/reversals rather than a fixed tile order.
            rotate = (session - 1) % len(order)
            order = order[rotate:] + order[:rotate]
            if session % 2 == 0:
                order.reverse()
            for tq, tk in order:
                def fn(tq=tq, tk=tk):
                    return launch_device(q, k, v, tile_q=tq, tile_k=tk)

                specialization = (tq, tk, case.D, case.Dv)
                cold_ms = None
                if specialization not in cold_calls:
                    start = time.perf_counter()
                    output = fn()
                    cp.cuda.get_current_stream().synchronize()
                    cold_ms = (time.perf_counter() - start) * 1000.0
                    cold_calls.add(specialization)
                    actual = cp.asnumpy(output)
                    if not passes_mixed_tolerance(
                        actual, expected, abs_scale=ABS_TOL, rel_scale=REL_TOL
                    ):
                        raise AssertionError(
                            f"{case.name} tile {tq}x{tk} failed reference gate before timing"
                        )
                else:
                    output = fn()
                    cp.cuda.get_current_stream().synchronize()
                    actual = cp.asnumpy(output)
                    if not passes_mixed_tolerance(
                        actual, expected, abs_scale=ABS_TOL, rel_scale=REL_TOL
                    ):
                        raise AssertionError(
                            f"{case.name} tile {tq}x{tk} failed reference gate before timing"
                        )

                timed = timed_cupy(
                    fn, cp, args.warmup, args.iterations, args.repeats_per_sample
                )
                spread_pct = (
                    100.0 * (timed["max_sample_ms"] - timed["min_sample_ms"]) /
                    timed["median_ms"] if timed["median_ms"] else None
                )
                all_trials.append({
                    "case": case.name, "B": case.B, "H": case.H, "N": case.N,
                    "D": case.D, "Dv": case.Dv, "seed": case.seed,
                    "tile_q": tq, "tile_k": tk, "session": session,
                    "tile_order": [f"{a}x{b}" for a, b in order],
                    "correctness_pass": True,
                    "first_call_ms": cold_ms,
                    "median_ms": timed["median_ms"], "p90_ms": timed["p90_ms"],
                    "min_sample_ms": timed["min_sample_ms"],
                    "max_sample_ms": timed["max_sample_ms"],
                    "sample_range_pct": spread_pct,
                    "samples_ms": timed["samples_ms"],
                    "warmup": args.warmup, "iterations": args.iterations,
                    "repeats_per_sample": args.repeats_per_sample,
                })
        del q, k, v
        cp.get_default_memory_pool().free_all_blocks()

    grouped = defaultdict(list)
    for row in all_trials:
        grouped[(row["case"], row["tile_q"], row["tile_k"])].append(row)

    summaries = []
    for case in cases:
        case_summaries = {}
        for (name, tq, tk), rows in sorted(grouped.items()):
            if name != case.name:
                continue
            times = [float(r["median_ms"]) for r in rows]
            center = float(statistics.median(times))
            range_pct = (
                100.0 * (max(times) - min(times)) / center if center else None
            )
            summary = {
                "case": name, "tile_q": tq, "tile_k": tk,
                "session_count": len(rows), "median_ms": center,
                "median_p90_ms": float(statistics.median(float(r["p90_ms"]) for r in rows)),
                "min_session_median_ms": min(times), "max_session_median_ms": max(times),
                "range_pct": range_pct, "all_correctness_pass": all(r["correctness_pass"] for r in rows),
                "first_call_ms": next((r["first_call_ms"] for r in rows if r["first_call_ms"] is not None), None),
            }
            summaries.append(summary)
            case_summaries[f"{tq}x{tk}"] = summary
        recommendation = recommend_tile(case_summaries, min_gain_pct=args.min_gain_pct)
        for summary in summaries:
            if summary["case"] == case.name:
                summary["recommended_tile_q"] = recommendation["tile_q"]
                summary["recommended_tile_k"] = recommendation["tile_k"]
                summary["recommendation_credible"] = recommendation["credible"]
                summary["recommendation_status"] = recommendation["status"]
                summary["gain_pct_vs_default"] = recommendation.get("gain_pct_vs_default")

    gpu = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu.get("name", b"")
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")
    payload = {
        "title": "Correctness-gated cuTile tile autotuning",
        "environment": {
            "gpu": str(gpu_name),
            "cupy": cp.__version__,
            "cuda_tile": __import__("importlib.metadata", fromlist=["version"]).version("cuda-tile"),
            "warmup": args.warmup,
            "iterations_per_session": args.iterations,
            "launches_per_event_sample": args.repeats_per_sample,
            "session_count": args.sessions,
            "min_gain_pct": args.min_gain_pct,
        },
        "method": "Every legal tile passes the independent FP32 NumPy oracle before timing. Three or more sessions rotate tile order; first-call timing is separate. A recommendation must beat the default by more than both the minimum gain threshold and half the combined session range.",
        "summary": summaries,
        "raw_trials": all_trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = [
        "case", "tile_q", "tile_k", "session_count", "median_ms",
        "median_p90_ms", "min_session_median_ms", "max_session_median_ms",
        "range_pct", "all_correctness_pass", "first_call_ms",
        "recommended_tile_q", "recommended_tile_k", "recommendation_credible",
        "recommendation_status", "gain_pct_vs_default",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in summaries)
    for case in cases:
        recommendation = next(row for row in summaries if row["case"] == case.name)
        print(
            f"{case.name}: recommendation={recommendation['recommended_tile_q']}x"
            f"{recommendation['recommended_tile_k']} credible={recommendation['recommendation_credible']} "
            f"status={recommendation['recommendation_status']}"
        )
    print(f"AUTOTUNE_JSON={args.output}")
    print(f"AUTOTUNE_CSV={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
