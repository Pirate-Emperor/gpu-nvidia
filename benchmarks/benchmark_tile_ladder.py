#!/usr/bin/env python3
"""Controlled A100 benchmark for v1/v2/v3/current shared-tile CUDA kernels."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "src" / "variants")]

from cases import MANDATORY_CASES
from shared_tile_v1 import _kernel as v1_kernel, launch_device as v1_launch
from shared_tile_v2 import _kernel as v2_kernel, launch_device as v2_launch
from shared_tile_v3 import _kernel as v3_kernel, launch_device as v3_launch
from shared_tile_cuda_attention import _kernel as cur_kernel, launch_device as cur_launch

TILES = (16, 32, 64)
ALL_CASES = tuple(c.name for c in MANDATORY_CASES)
# v1 is intentionally included only on small/medium cases because its
# one-row/two-pass structure makes very long cases prohibitively expensive.
V1_CASES = ("short_d32", "short_d64", "short_multihead_d64", "medium_d64")
STEADY_CASES = ALL_CASES


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
        s = cp.cuda.Event()
        e = cp.cuda.Event()
        s.record(stream)
        fn()
        e.record(stream)
        e.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(s, e)))
    x = np.asarray(samples, dtype=np.float64)
    return float(np.median(x)), float(np.percentile(x, 90))


def cold_compile(kernel_fn, tile_k, cp):
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    t0 = time.perf_counter()
    k = kernel_fn(tile_k)
    k.compile()
    stream.synchronize()
    return float((time.perf_counter() - t0) * 1000.0)


def throughput_tflops(case, median_ms):
    pairs = case.N * (case.N + 1) // 2
    flops = 2.0 * case.B * case.H * pairs * (case.D + case.Dv)
    return flops / (median_ms * 1.0e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--output", type=Path, default=Path("results/tile_ladder.json"))
    args = ap.parse_args()
    if args.warmup < 1 or args.iterations < 30:
        ap.error("warmup >= 1 and iterations >= 30 required")

    import cupy as cp

    cp.cuda.Device(0).use()
    dev = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(dev, bytes):
        dev = dev.decode(errors="replace")

    versions = {
        "v1": (v1_kernel, v1_launch, set(V1_CASES)),
        "v2": (v2_kernel, v2_launch, set(ALL_CASES)),
        "v3": (v3_kernel, v3_launch, set(ALL_CASES)),
        "current": (cur_kernel, cur_launch, set(ALL_CASES)),
    }

    rows = []

    # Compile every version/tile once in a fresh CuPy cache and record that
    # cold compile separately from steady-state execution.
    # The kernel source is identical across tile_k; tile_k is a runtime
    # argument. Only the first compile per version is a true cold JIT
    # measurement in the fresh CuPy cache.
    cold_ms = {}
    for version, (kernel_fn, _, cases) in versions.items():
        cold_ms[version] = cold_compile(kernel_fn, TILES[0], cp)

    for name in ALL_CASES:
        case = next(c for c in MANDATORY_CASES if c.name == name)
        q, k, v = make_inputs(case, cp)
        for version, (_, launch_fn, cases) in versions.items():
            if name not in cases:
                continue
            for tile_k in TILES:
                med, p90 = timed(
                    lambda f=launch_fn, tk=tile_k: f(q, k, v, tile_k=tk),
                    cp,
                    args.warmup,
                    args.iterations,
                )
                rows.append({
                    "case": name,
                    "B": case.B,
                    "H": case.H,
                    "N": case.N,
                    "D": case.D,
                    "Dv": case.Dv,
                    "tile_k": tile_k,
                    "version": version,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "cold_compile_ms": cold_ms[version] if tile_k == TILES[0] else None,
                    "median_ms": med,
                    "p90_ms": p90,
                    "throughput_tflops": throughput_tflops(case, med),
                })
        del q, k, v
        cp.get_default_memory_pool().free_all_blocks()

    by_key = {(r["case"], r["tile_k"]): {} for r in rows}
    for r in rows:
        by_key[(r["case"], r["tile_k"])][r["version"]] = r["median_ms"]
    for r in rows:
        peer = by_key[(r["case"], r["tile_k"])]
        r["speedup_vs_v1"] = peer["v1"] / r["median_ms"] if "v1" in peer else None
        r["speedup_vs_v2"] = peer["v2"] / r["median_ms"] if "v2" in peer else None
        r["speedup_vs_v3"] = peer["v3"] / r["median_ms"] if "v3" in peer else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": str(dev),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tiles": list(TILES),
        "cold_compile_tile_k": TILES[0],
        "v1_cases": list(V1_CASES),
        "cases": list(ALL_CASES),
        "results": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("COLD_JIT")
    for version in versions:
        print(f"  {version} tk={TILES[0]} compile={cold_ms[version]:.3f}ms")

    for name in ALL_CASES:
        print(f"CASE {name}")
        for tile_k in TILES:
            vals = [r for r in rows if r["case"] == name and r["tile_k"] == tile_k]
            msg = []
            for r in vals:
                msg.append(f'{r["version"]}={r["median_ms"]:.3f}ms')
            print(f"  tk={tile_k:2d} " + " ".join(msg))


if __name__ == "__main__":
    raise SystemExit(main())
