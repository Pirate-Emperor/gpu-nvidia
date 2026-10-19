#!/usr/bin/env python3
"""Compare genuine cuTile attention with CUDA PyTorch SDPA on the native GPU.

Outputs JSON/CSV with per-case correctness, first-call wall time, steady-state
CUDA-event median/P90 and detected SDPA operator/backend. Torch is only an
optional reference; this script does not alter the main cuTile implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from cases import MANDATORY_CASES
from cutile_attention import launch_device as cutile_launch_device
from reference import causal_attention_fp32

TILES = {
    "short_d32": (64, 64),
    "short_d64": (64, 64),
    "short_multihead_d64": (64, 64),
    "medium_d64": (64, 64),
    "medium_d128": (64, 64),
    "long_d128": (64, 64),
    "longer_batched_d128": (64, 128),
}


def error_summary(actual, reference):
    a = np.asarray(actual, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    error = np.abs(a - r)
    allowed = 2e-2 + 2e-2 * np.abs(r)
    mask = np.abs(r) >= 1e-2
    rel = error[mask] / np.abs(r[mask]) if np.any(mask) else np.asarray([0.0])
    violations = int(np.count_nonzero(error > allowed))
    return {
        "pass": bool(violations == 0 and np.isfinite(a).all()),
        "max_abs_error": float(error.max(initial=0.0)),
        "max_rel_error_abs_ref_ge_1e_2": float(rel.max(initial=0.0)),
        "max_error_over_allowed_tolerance": float((error / allowed).max(initial=0.0)),
        "tolerance_violations": violations,
    }


def make_inputs(case):
    rng = np.random.default_rng(case.seed)
    q = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32).astype(np.float16)
    return q, k, v


def time_cupy(fn, cp, warmup, iterations, repeats):
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
    return {
        "median_ms": float(np.median(samples)),
        "p90_ms": float(np.percentile(samples, 90)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "sample_count": len(samples),
        "repeats_per_sample": repeats,
    }


def time_torch(fn, torch, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) / repeats)
    return {
        "median_ms": float(np.median(samples)),
        "p90_ms": float(np.percentile(samples, 90)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "sample_count": len(samples),
        "repeats_per_sample": repeats,
    }


def detect_sdpa_operator(query, key, value, torch, functional):
    """Observe the selected operator without inferring backend from timing."""
    from torch.profiler import profile, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        torch.cuda.synchronize()
    names = sorted({
        event.key for event in prof.key_averages()
        if "scaled_dot_product" in event.key.lower()
        or "flash_attention" in event.key.lower()
        or "efficient_attention" in event.key.lower()
        or "cudnn_attention" in event.key.lower()
    })
    return out, names


def timed_supported_backends(query, key, value, torch, functional, warmup, iterations, repeats):
    """Time forced backends on the long/representative cases; preserve each failure."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    candidates = [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("MATH", SDPBackend.MATH),
    ]
    rows = []
    for name, backend in candidates:
        if backend is None:
            rows.append({"backend": name, "available": False, "reason": "enum unavailable in this PyTorch build"})
            continue

        def fn():
            with sdpa_kernel(backends=[backend]):
                return functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True
                )

        try:
            out = fn()
            torch.cuda.synchronize()
            timing = time_torch(fn, torch, warmup, iterations, repeats)
            rows.append({"backend": name, "available": True, **timing})
        except Exception as exc:
            torch.cuda.synchronize()
            rows.append({
                "backend": name,
                "available": False,
                "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "framework-attention")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    parser.add_argument("--force-backends-on", default="medium_d128,long_d128,longer_batched_d128")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1:
        parser.error("require warmup >= 1, iterations >= 30, repeats-per-sample >= 1")

    try:
        import cupy as cp
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        print(f"ERROR: PyTorch/CuPy import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA-enabled PyTorch is installed but torch.cuda.is_available() is False", file=sys.stderr)
        return 3
    if not cp.cuda.is_available():
        print("ERROR: CuPy cannot access a CUDA device", file=sys.stderr)
        return 4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    force_names = {item.strip() for item in args.force_backends_on.split(",") if item.strip()}
    device = torch.cuda.get_device_properties(0)
    results = {
        "title": "cuTile vs CUDA PyTorch SDPA",
        "method": {
            "warmup_iterations": args.warmup,
            "cuda_event_samples": args.iterations,
            "kernel_calls_averaged_per_sample": args.repeats_per_sample,
            "steady_state_samples_per_config": args.iterations,
            "reference": "independent full-matrix FP32 NumPy causal attention",
            "tolerance": "abs(actual-reference) <= 0.02 + 0.02*abs(reference)",
            "pytorch_input_dtype": "torch.float16",
            "pytorch_api": "torch.nn.functional.scaled_dot_product_attention(is_causal=True, dropout_p=0)",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": device.name,
            "compute_capability": f"{device.major}.{device.minor}",
            "pytorch_flash_attention_built": bool(torch.backends.cuda.is_flash_attention_available()),
        },
        "cases": [],
    }

    for case in MANDATORY_CASES:
        tq, tk = TILES[case.name]
        q_np, k_np, v_np = make_inputs(case)
        reference = causal_attention_fp32(q_np, k_np, v_np, output_dtype=np.float32)

        q_cp, k_cp, v_cp = cp.asarray(q_np), cp.asarray(k_np), cp.asarray(v_np)
        def cutile_fn():
            return cutile_launch_device(q_cp, k_cp, v_cp, tile_q=tq, tile_k=tk)

        t0 = time.perf_counter()
        cutile_first = cutile_fn()
        cp.cuda.get_current_stream().synchronize()
        cutile_first_ms = (time.perf_counter() - t0) * 1000.0
        cutile_actual = cp.asnumpy(cutile_first)
        cutile_error = error_summary(cutile_actual, reference)
        if not cutile_error["pass"]:
            raise AssertionError(f"{case.name}: cuTile output fails independent reference tolerance")
        cutile_timing = time_cupy(cutile_fn, cp, args.warmup, args.iterations, args.repeats_per_sample)

        q_t = torch.from_numpy(q_np.copy()).to(device="cuda", dtype=torch.float16).contiguous()
        k_t = torch.from_numpy(k_np.copy()).to(device="cuda", dtype=torch.float16).contiguous()
        v_t = torch.from_numpy(v_np.copy()).to(device="cuda", dtype=torch.float16).contiguous()
        torch_fn = lambda: F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        t0 = time.perf_counter()
        torch_first = torch_fn()
        torch.cuda.synchronize()
        torch_first_ms = (time.perf_counter() - t0) * 1000.0
        torch_actual = torch_first.detach().cpu().numpy()
        torch_error = error_summary(torch_actual, reference)
        if not torch_error["pass"]:
            raise AssertionError(f"{case.name}: PyTorch SDPA output fails independent reference tolerance")
        _, selected_ops = detect_sdpa_operator(q_t, k_t, v_t, torch, F)
        torch_timing = time_torch(torch_fn, torch, args.warmup, args.iterations, args.repeats_per_sample)

        row = {
            "name": case.name,
            "shape": [case.B, case.H, case.N, case.D],
            "tile": [tq, tk],
            "reference_dtype": "float32",
            "cutile": {
                "first_call_wall_ms": cutile_first_ms,
                "steady_state": cutile_timing,
                "error_vs_reference": cutile_error,
            },
            "pytorch_sdpa_auto": {
                "first_call_wall_ms": torch_first_ms,
                "steady_state": torch_timing,
                "error_vs_reference": torch_error,
                "selected_operator_names": selected_ops,
                "median_speedup_cutile_over_sdpa": torch_timing["median_ms"] / cutile_timing["median_ms"],
            },
        }
        if case.name in force_names:
            row["forced_backend_trials"] = timed_supported_backends(
                q_t, k_t, v_t, torch, F, args.warmup, args.iterations, args.repeats_per_sample
            )
        results["cases"].append(row)
        torch.cuda.synchronize()
        cp.cuda.get_current_stream().synchronize()
        print(
            f"CASE={case.name} shape={row['shape']} tile={tq}x{tk} "
            f"cuTile={cutile_timing['median_ms']:.6f}ms "
            f"SDPA={torch_timing['median_ms']:.6f}ms "
            f"SDPA_OPS={';'.join(selected_ops)} "
            f"cuTile_ORACLE={'PASS' if cutile_error['pass'] else 'FAIL'} "
            f"SDPA_ORACLE={'PASS' if torch_error['pass'] else 'FAIL'}"
        )

    out_json = args.out_dir / "framework-attention-results.json"
    out_csv = args.out_dir / "framework-attention-results.csv"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    rows = []
    for item in results["cases"]:
        rows.append({
            "case": item["name"],
            "B": item["shape"][0], "H": item["shape"][1], "N": item["shape"][2], "D": item["shape"][3],
            "tile_q": item["tile"][0], "tile_k": item["tile"][1],
            "cutile_first_call_wall_ms": item["cutile"]["first_call_wall_ms"],
            "cutile_median_ms": item["cutile"]["steady_state"]["median_ms"],
            "cutile_p90_ms": item["cutile"]["steady_state"]["p90_ms"],
            "cutile_max_abs_error": item["cutile"]["error_vs_reference"]["max_abs_error"],
            "cutile_oracle_pass": item["cutile"]["error_vs_reference"]["pass"],
            "sdpa_first_call_wall_ms": item["pytorch_sdpa_auto"]["first_call_wall_ms"],
            "sdpa_median_ms": item["pytorch_sdpa_auto"]["steady_state"]["median_ms"],
            "sdpa_p90_ms": item["pytorch_sdpa_auto"]["steady_state"]["p90_ms"],
            "sdpa_selected_ops": ";".join(item["pytorch_sdpa_auto"]["selected_operator_names"]),
            "sdpa_max_abs_error": item["pytorch_sdpa_auto"]["error_vs_reference"]["max_abs_error"],
            "sdpa_oracle_pass": item["pytorch_sdpa_auto"]["error_vs_reference"]["pass"],
            "cutile_over_sdpa_speedup": item["pytorch_sdpa_auto"]["median_speedup_cutile_over_sdpa"],
        })
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"FRAMEWORK_REFERENCE=PASS; cases={len(results['cases'])}; JSON={out_json}; CSV={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
