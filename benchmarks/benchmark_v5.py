#!/usr/bin/env python3
"""Benchmark v5 half2 QK against current shared-tile CUDA."""

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
    xs = []
    for _ in range(iterations):
        s = cp.cuda.Event()
        e = cp.cuda.Event()
        s.record(stream)
        fn()
        e.record(stream)
        e.synchronize()
        xs.append(float(cp.cuda.get_elapsed_time(s, e)))
    a = np.asarray(xs, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--output", type=Path, default=Path("results/v5_experiment.json"))
    args = ap.parse_args()
    if args.warmup < 1 or args.iterations < 30:
        ap.error("warmup >= 1 and iterations >= 30 required")

    import cupy as cp
    cp.cuda.Device(0).use()

    rows = []
    for case in MANDATORY_CASES:
        q, k, v = make_inputs(case, cp)
        for version, fn in (("current", cur_launch), ("v5", v5_launch)):
            for tile_k in TILES:
                med, p90 = timed(
                    lambda f=fn, tk=tile_k: f(q, k, v, tile_k=tk),
                    cp, args.warmup, args.iterations
                )
                rows.append({
                    "case": case.name,
                    "B": case.B,
                    "H": case.H,
                    "N": case.N,
                    "D": case.D,
                    "tile_k": tile_k,
                    "version": version,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "median_ms": med,
                    "p90_ms": p90,
                })
        del q, k, v
        cp.get_default_memory_pool().free_all_blocks()

    pivot = {}
    for r in rows:
        pivot.setdefault((r["case"], r["tile_k"]), {})[r["version"]] = r["median_ms"]
    for r in rows:
        r["current_over_v5"] = pivot[(r["case"], r["tile_k"])]["current"] / r["median_ms"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "gpu": str(cp.cuda.runtime.getDeviceProperties(0)["name"]),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tiles": list(TILES),
        "cases": [c.name for c in MANDATORY_CASES],
        "results": rows,
    }, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for case in MANDATORY_CASES:
        for tk in TILES:
            p = pivot[(case.name, tk)]
            print(
                f"{case.name:24s} tk={tk:2d} "
                f"current={p['current']:.3f}ms v5={p['v5']:.3f}ms "
                f"current/v5={p['current']/p['v5']:.3f}x"
            )


if __name__ == "__main__":
    raise SystemExit(main())
