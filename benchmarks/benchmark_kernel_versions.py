#!/usr/bin/env python3
"""Benchmark preserved shared-tile CUDA kernel versions under identical conditions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "variants"))

from cases import MANDATORY_CASES
from shared_tile_v1 import launch_device as v1_launch
from shared_tile_v2 import launch_device as v2_launch


CASES = ("short_d64", "short_multihead_d64", "medium_d128")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--output", type=Path, default=Path("results/kernel_version_benchmark.json"))
    args = ap.parse_args()
    if args.warmup < 1 or args.iterations < 30:
        ap.error("warmup >= 1 and iterations >= 30 required")

    import cupy as cp

    by_name = {c.name: c for c in MANDATORY_CASES}
    rows = []
    for name in CASES:
        case = by_name[name]
        q, k, v = make_inputs(case, cp)
        for version, fn in (("v1", v1_launch), ("v2", v2_launch)):
            tile_k = 64
            med, p90 = timed(lambda f=fn: f(q, k, v, tile_k=tile_k), cp, args.warmup, args.iterations)
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
    for row in rows:
        pivot.setdefault(row["case"], {})[row["version"]] = row["median_ms"]
    for row in rows:
        other = "v2" if row["version"] == "v1" else "v1"
        row["speedup_vs_other"] = pivot[row["case"]][other] / row["median_ms"]

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"]
    if isinstance(dev, bytes):
        dev = dev.decode(errors="replace")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gpu": dev, "results": rows}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for case in CASES:
        p = pivot[case]
        print(f"{case:24s} v1={p['v1']:.3f} ms v2={p['v2']:.3f} ms v1/v2={p['v1']/p['v2']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
