#!/usr/bin/env python3
"""Create per-case cuTile accuracy diagnostics against the independent FP32 oracle.

This is a GPU experiment/report, not a replacement for pytest. It reruns every
divisible mandatory case/tile configuration and records element-wise errors.
All generated inputs stay in memory; only the JSON report is written.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cases import MANDATORY_CASES
from cutile_attention import forward
from reference import causal_attention_fp32

TILE_CONFIGS = ((32, 64), (64, 64), (64, 128), (128, 64), (128, 128))
ABS_SCALE = 2e-2
REL_SCALE = 2e-2
RELATIVE_ERROR_REFERENCE_FLOOR = 1e-2


def make_inputs(case):
    rng = np.random.default_rng(case.seed)
    q = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32).astype(np.float16)
    return q, k, v


def git_commit() -> str | None:
    # In Actions, GITHUB_SHA is the immutable triggering revision and is the
    # canonical provenance field. Retain the locally observed HEAD separately
    # because self-hosted runner path/junction layouts can expose another repo.
    workflow_sha = os.environ.get("GITHUB_SHA")
    if workflow_sha:
        return workflow_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def observed_git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "cutile-accuracy.json",
    )
    args = parser.parse_args()

    import cupy as cp
    import cuda.tile  # noqa: F401

    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props.get("name", b"<unknown>")
    gpu_name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)

    rows = []
    for case in MANDATORY_CASES:
        q, k, v = make_inputs(case)
        expected = causal_attention_fp32(q, k, v, output_dtype=np.float16)
        for tile_q, tile_k in TILE_CONFIGS:
            if case.N % tile_q or case.N % tile_k:
                continue

            actual = forward(q, k, v, tile_q=tile_q, tile_k=tile_k)
            if actual.shape != expected.shape:
                raise AssertionError(
                    f"{case.name} tile={tile_q}x{tile_k}: shape {actual.shape} != {expected.shape}"
                )
            if actual.dtype != np.float16:
                raise AssertionError(
                    f"{case.name} tile={tile_q}x{tile_k}: output dtype {actual.dtype} != float16"
                )
            if not np.isfinite(actual).all():
                raise AssertionError(f"{case.name} tile={tile_q}x{tile_k}: output has NaN/Inf")

            actual32 = actual.astype(np.float32)
            expected32 = expected.astype(np.float32)
            abs_diff = np.abs(actual32 - expected32)
            allowed = ABS_SCALE + REL_SCALE * np.abs(expected32)
            ratio = abs_diff / allowed
            violations = int(np.count_nonzero(abs_diff > allowed))
            max_abs = float(abs_diff.max(initial=0.0))
            rmse = float(np.sqrt(np.mean(np.square(abs_diff), dtype=np.float64)))
            p99_abs = float(np.percentile(abs_diff, 99))
            max_ratio = float(ratio.max(initial=0.0))
            relative_mask = np.abs(expected32) >= RELATIVE_ERROR_REFERENCE_FLOOR
            relative_count = int(np.count_nonzero(relative_mask))
            if relative_count:
                relative_errors = abs_diff[relative_mask] / np.abs(expected32[relative_mask])
                max_rel_nonzero = float(relative_errors.max(initial=0.0))
            else:
                max_rel_nonzero = None
            passed = violations == 0

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
                "elements": int(actual.size),
                "output_dtype": str(actual.dtype),
                "finite": True,
                "max_abs_error": max_abs,
                "p99_abs_error": p99_abs,
                "rmse_abs_error": rmse,
                "relative_error_reference_floor": RELATIVE_ERROR_REFERENCE_FLOOR,
                "relative_error_element_count": relative_count,
                "max_relative_error_for_abs_reference_ge_floor": max_rel_nonzero,
                "max_tolerance_ratio": max_ratio,
                "tolerance_violation_count": violations,
                "tolerance": f"abs(error) <= {ABS_SCALE} + {REL_SCALE} * abs(reference)",
                "passed": passed,
            }
            rows.append(row)
            print(
                f"{case.name:24s} tile={tile_q}x{tile_k} "
                f"max_abs={max_abs:.7g} p99_abs={p99_abs:.7g} "
                f"rmse={rmse:.7g} max_tol_ratio={max_ratio:.6f} "
                f"violations={violations}"
            )
            if not passed:
                raise AssertionError(
                    f"{case.name} tile={tile_q}x{tile_k}: {violations} tolerance violations; "
                    f"max_abs={max_abs}, max_tolerance_ratio={max_ratio}"
                )

    report = {
        "experiment": "cuTile accuracy against independent FP32 NumPy oracle",
        "git_commit": git_commit(),
        "git_head_observed": observed_git_head(),
        "git_commit_source": "GITHUB_SHA" if os.environ.get("GITHUB_SHA") else "git rev-parse HEAD",
        "gpu": gpu_name,
        "cuda_driver_api_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_api_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_tile": importlib.metadata.version("cuda-tile"),
        "required_tolerance": {"absolute_scale": ABS_SCALE, "relative_scale": REL_SCALE},
        "case_count": len(rows),
        "all_passed": all(row["passed"] for row in rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ACCURACY_REPORT={args.output}")
    print(f"ACCURACY_CASES={len(rows)}")
    print(f"ACCURACY_ALL_PASSED={report['all_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
