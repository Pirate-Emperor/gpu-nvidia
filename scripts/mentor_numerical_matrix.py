#!/usr/bin/env python3
"""Run the 29 legal cuTile attention case/tile pairs against an independent NumPy reference.

This is a numerical validation matrix, not a timing benchmark. It reports one
row per legal specialization and exits non-zero on any contract/tolerance failure.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cases import MANDATORY_CASES
from cutile_attention import launch_device
from reference import causal_attention_fp32, passes_mixed_tolerance

TILES = ((32, 64), (64, 64), (64, 128), (128, 64), (128, 128))
ABS_TOL = 2e-2
REL_TOL = 2e-2
EXPECTED_LEGAL_PAIRS = 29


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every legal mandatory case/tile specialization on actual cuTile."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/logs/mentor_legal_numerical_matrix.json"),
        help="JSON path; a sibling CSV is also written",
    )
    parser.add_argument(
        "--case",
        choices=[case.name for case in MANDATORY_CASES],
        help="restrict the matrix to one named workload",
    )
    parser.add_argument("--tile-q", type=int, help="query tile for one configuration")
    parser.add_argument("--tile-k", type=int, help="key tile for one configuration")
    args = parser.parse_args()
    selecting_tile = args.tile_q is not None or args.tile_k is not None
    if selecting_tile and (
        args.case is None or args.tile_q is None or args.tile_k is None
    ):
        parser.error("--case, --tile-q and --tile-k must be supplied together for one configuration")

    selected_cases = tuple(
        case for case in MANDATORY_CASES
        if args.case is None or case.name == args.case
    )
    selected_tiles = tuple(
        (tile_q, tile_k) for tile_q, tile_k in TILES
        if not selecting_tile or (tile_q == args.tile_q and tile_k == args.tile_k)
    )
    if selecting_tile and not selected_tiles:
        parser.error(f"tile {args.tile_q}x{args.tile_k} is not one of the five defined candidates")
    legal_selected_pairs = [
        (case, tile_q, tile_k)
        for case in selected_cases
        for tile_q, tile_k in selected_tiles
        if case.N % tile_q == 0 and case.N % tile_k == 0
    ]
    if selecting_tile and len(legal_selected_pairs) != 1:
        parser.error(
            f"{args.case} at N={selected_cases[0].N} with tile "
            f"{args.tile_q}x{args.tile_k} is not a legal configuration"
        )
    expected_pair_count = (
        EXPECTED_LEGAL_PAIRS
        if args.case is None and not selecting_tile
        else len(legal_selected_pairs)
    )

    try:
        import cupy as cp
        import cuda.tile  # noqa: F401
    except Exception as exc:
        print(
            f"ERROR: actual CuPy/cuTile runtime is required: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props.get("name", b"")
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode(errors="replace")

    rows = []
    checked_pairs = 0
    print(f"GPU: {gpu_name}")
    print(f"Reference: independent full-matrix FP32 NumPy attention")
    print(f"Mixed tolerance: abs(actual - reference) <= {ABS_TOL} + {REL_TOL} * abs(reference)")
    print("Only positive power-of-two tiles that divide N are tested.")
    if args.case is not None:
        print(f"FILTER_CASE={args.case}")
    if selecting_tile:
        print(f"FILTER_TILE={args.tile_q}x{args.tile_k}")
    print("")
    print(f"{'CASE':25s} {'SHAPE (B,H,N,D)':22s} {'TILE':9s} {'MAX ABS':>12s} {'P99 ABS':>12s} {'RMSE':>12s} {'TOL ENV %':>11s} STATUS")

    for case in selected_cases:
        q_np, k_np, v_np = make_inputs(case)
        expected = causal_attention_fp32(q_np, k_np, v_np, output_dtype=np.float16)
        q_gpu, k_gpu, v_gpu = cp.asarray(q_np), cp.asarray(k_np), cp.asarray(v_np)

        for tile_q, tile_k in selected_tiles:
            if case.N % tile_q or case.N % tile_k:
                continue

            checked_pairs += 1
            row = {
                "case": case.name,
                "B": case.B,
                "H": case.H,
                "N": case.N,
                "D": case.D,
                "Dv": case.Dv,
                "seed": case.seed,
                "tile_q": tile_q,
                "tile_k": tile_k,
                "abs_tolerance": ABS_TOL,
                "rel_tolerance": REL_TOL,
                "max_abs_error": None,
                "p99_abs_error": None,
                "rmse": None,
                "max_relative_error_ref_abs_ge_0_01": None,
                "max_tolerance_envelope_pct": None,
                "status": "FAIL",
                "error": None,
            }
            try:
                actual_gpu = launch_device(
                    q_gpu, k_gpu, v_gpu, tile_q=tile_q, tile_k=tile_k
                )
                cp.cuda.get_current_stream().synchronize()
                actual = cp.asnumpy(actual_gpu)

                if actual.shape != expected.shape:
                    raise AssertionError(
                        f"shape mismatch actual={actual.shape} expected={expected.shape}"
                    )
                if actual.dtype != np.float16:
                    raise AssertionError(f"output dtype {actual.dtype}; expected float16")
                if not np.isfinite(actual).all():
                    raise AssertionError("output contains NaN or infinity")

                a = actual.astype(np.float64)
                e = expected.astype(np.float64)
                delta = np.abs(a - e)
                allowed = ABS_TOL + REL_TOL * np.abs(e)
                rel_mask = np.abs(e) >= 0.01
                rel_max = (
                    float(np.max(delta[rel_mask] / np.abs(e[rel_mask])))
                    if np.any(rel_mask)
                    else None
                )
                passed = passes_mixed_tolerance(
                    actual, expected, abs_scale=ABS_TOL, rel_scale=REL_TOL
                )
                row.update(
                    max_abs_error=float(np.max(delta, initial=0.0)),
                    p99_abs_error=float(np.percentile(delta, 99)),
                    rmse=float(np.sqrt(np.mean(np.square(a - e)))),
                    max_relative_error_ref_abs_ge_0_01=rel_max,
                    max_tolerance_envelope_pct=float(
                        100.0 * np.max(delta / allowed, initial=0.0)
                    ),
                    status="PASS" if passed else "FAIL",
                )
                if not passed:
                    row["error"] = "mixed-tolerance check failed"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"

            rows.append(row)
            shape = f"({case.B},{case.H},{case.N},{case.D})"
            vals = (
                f"{row['max_abs_error']:.7g}" if row["max_abs_error"] is not None else "n/a",
                f"{row['p99_abs_error']:.7g}" if row["p99_abs_error"] is not None else "n/a",
                f"{row['rmse']:.7g}" if row["rmse"] is not None else "n/a",
                f"{row['max_tolerance_envelope_pct']:.4f}" if row["max_tolerance_envelope_pct"] is not None else "n/a",
            )
            print(
                f"{case.name:25s} {shape:22s} {tile_q}x{tile_k:<5d} "
                f"{vals[0]:>12s} {vals[1]:>12s} {vals[2]:>12s} {vals[3]:>11s} {row['status']}"
            )
            if row["error"]:
                print(f"  ERROR: {row['error']}")

        del q_gpu, k_gpu, v_gpu, q_np, k_np, v_np, expected
        cp.get_default_memory_pool().free_all_blocks()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    metadata = {
        "title": "Actual cuTile legal case/tile numerical matrix",
        "gpu": str(gpu_name),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cupy": cp.__version__,
        "cuda_tile": importlib.metadata.version("cuda-tile"),
        "reference": "independent full-matrix FP32 NumPy attention",
        "tolerance": "abs(actual-reference) <= 0.02 + 0.02*abs(reference)",
        "legal_pair_count": checked_pairs,
        "expected_legal_pair_count": expected_pair_count,
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] != "PASS" for row in rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    passed = metadata["pass_count"]
    failed = metadata["fail_count"]
    print("")
    print(f"LEGAL_PAIRS={checked_pairs} EXPECTED={expected_pair_count}")
    print(f"NUMERICAL_MATRIX={passed} passed / {failed} failed")
    print(f"JSON={args.output}")
    print(f"CSV={csv_path}")

    if checked_pairs != expected_pair_count:
        print(
            f"ERROR: legal pair count drifted; expected {expected_pair_count}, got {checked_pairs}",
            file=sys.stderr,
        )
        return 1
    if failed:
        return 1
    print("NUMERICAL_MATRIX=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
