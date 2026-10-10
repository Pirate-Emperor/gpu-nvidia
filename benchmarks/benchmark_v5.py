#!/usr/bin/env python3
"""Paired, order-alternating benchmark of v5 half2 QK vs current shared-tile CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "src" / "variants")]

from cases import MANDATORY_CASES
from shared_tile_v5 import launch_device as v5_launch
from shared_tile_cuda_attention import launch_device as cur_launch

TILES = (16, 32, 64)


def make_inputs(case, cp):
    rng = np.random.default_rng(case.seed)
    q = cp.asarray(rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16))
    k = cp.asarray(rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16))
    v = cp.asarray(rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32).astype(np.float16))
    return q, k, v


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
    vals = np.asarray(samples, dtype=np.float64)
    return float(np.median(vals)), float(np.percentile(vals, 90))


def decode(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def summarize(rows, repeats):
    summary = []
    for case in (c.name for c in MANDATORY_CASES):
        for tile_k in TILES:
            by_ver = {
                ver: [r for r in rows if r["case"] == case and r["tile_k"] == tile_k and r["version"] == ver]
                for ver in ("current", "v5")
            }
            if any(len(v) != repeats for v in by_ver.values()):
                raise RuntimeError(f"incomplete paired measurements: {case=} {tile_k=} {by_ver.keys()=}")
            cur = {r["repeat"]: r["median_ms"] for r in by_ver["current"]}
            v5 = {r["repeat"]: r["median_ms"] for r in by_ver["v5"]}
            ratios = [cur[i] / v5[i] for i in range(repeats)]
            summary.append({
                "case": case,
                "tile_k": tile_k,
                "repeats": repeats,
                "current_median_ms": float(np.median(list(cur.values()))),
                "current_p90_median_ms": float(np.median([r["p90_ms"] for r in by_ver["current"]])),
                "v5_median_ms": float(np.median(list(v5.values()))),
                "v5_p90_median_ms": float(np.median([r["p90_ms"] for r in by_ver["v5"]])),
                "paired_speedup_current_over_v5": float(np.median(ratios)),
                "paired_speedup_min": float(np.min(ratios)),
                "paired_speedup_max": float(np.max(ratios)),
            })
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3, help="paired sessions per case/tile; order alternates")
    ap.add_argument("--output", type=Path, default=Path("results/v5_experiment.json"))
    args = ap.parse_args()
    if args.warmup < 1 or args.iterations < 30 or args.repeats < 2:
        ap.error("warmup >= 1, iterations >= 30, and repeats >= 2 required")

    import cupy as cp

    cp.cuda.Device(0).use()
    dev = cp.cuda.runtime.getDeviceProperties(0)
    device_name = decode(dev["name"])
    rows = []
    funcs = {"current": cur_launch, "v5": v5_launch}

    # Paired trials share the same resident Q/K/V data. The first-measured
    # version alternates by repeat and tile to reduce order/load bias.
    for case in MANDATORY_CASES:
        q, k, v = make_inputs(case, cp)
        for tile_index, tile_k in enumerate(TILES):
            for repeat in range(args.repeats):
                order = ("current", "v5") if (repeat + tile_index) % 2 == 0 else ("v5", "current")
                for order_index, version in enumerate(order):
                    med, p90 = timed(
                        lambda f=funcs[version], tk=tile_k: f(q, k, v, tile_k=tk),
                        cp, args.warmup, args.iterations
                    )
                    rows.append({
                        "case": case.name,
                        "B": case.B,
                        "H": case.H,
                        "N": case.N,
                        "D": case.D,
                        "Dv": case.Dv,
                        "tile_k": tile_k,
                        "version": version,
                        "repeat": repeat,
                        "order_in_pair": order_index,
                        "warmup": args.warmup,
                        "iterations": args.iterations,
                        "median_ms": med,
                        "p90_ms": p90,
                    })
        del q, k, v
        cp.get_default_memory_pool().free_all_blocks()

    summary = summarize(rows, args.repeats)
    props = cp.cuda.runtime.getDeviceProperties(0)
    try:
        driver_ver = cp.cuda.runtime.driverGetVersion()
    except Exception:
        driver_ver = None
    try:
        runtime_ver = cp.cuda.runtime.runtimeGetVersion()
    except Exception:
        runtime_ver = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "paired_current_vs_v5_order_alternating",
        "gpu": device_name,
        "compute_capability": f"{props.get('major', '?')}.{props.get('minor', '?')}",
        "driver_version": driver_ver,
        "cuda_runtime_version": runtime_ver,
        "cupy_version": cp.__version__,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "tiles": list(TILES),
        "cases": [c.name for c in MANDATORY_CASES],
        "timing_notes": "Each median/P90 uses CUDA events after warmup. Current/v5 order alternates per paired repeat. Summary reports median of session medians and median paired current/v5 ratios; speedup > 1 means v5 was faster.",
        "summary": summary,
        "raw_sessions": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with args.output.with_name(args.output.stem + "_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    for r in summary:
        print(
            f'{r["case"]:24s} tk={r["tile_k"]:2d} '
            f'current={r["current_median_ms"]:.3f}ms '
            f'v5={r["v5_median_ms"]:.3f}ms '
            f'paired current/v5={r["paired_speedup_current_over_v5"]:.3f}x '
            f'range=[{r["paired_speedup_min"]:.3f}, {r["paired_speedup_max"]:.3f}]'
        )


if __name__ == "__main__":
    raise SystemExit(main())
