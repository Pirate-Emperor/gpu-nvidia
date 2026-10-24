#!/usr/bin/env python3
"""Compare cuTile with CUDA PyTorch SDPA. Check both outputs against NumPy and record the backend and cold/steady timings."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from baseline import launch_device as baseline_launch_device
from cases import MANDATORY_CASES
from cutile_attention import launch_device as cutile_launch_device
from reference import causal_attention_fp32, max_abs_error, max_rel_error, passes_mixed_tolerance

# Case and tile pairs are frozen, legal, and span all required D values.
CONFIGS = (
    ("short_d32", 64, 64),
    ("short_d64", 64, 64),
    ("short_multihead_d64", 64, 64),
    ("medium_d64", 64, 64),
    ("medium_d128", 64, 64),
    ("medium_d128", 64, 128),
    ("long_d128", 64, 64),
    ("long_d128", 64, 128),
    ("longer_batched_d128", 64, 64),
    ("longer_batched_d128", 64, 128),
)
ABS_TOL = 2e-2
REL_TOL = 2e-2


def make_inputs(case, cp):
    rng = np.random.default_rng(case.seed)
    q = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    k = rng.standard_normal((case.B, case.H, case.N, case.D), dtype=np.float32).astype(np.float16)
    v = rng.standard_normal((case.B, case.H, case.N, case.Dv), dtype=np.float32).astype(np.float16)
    return q, k, v, cp.asarray(q), cp.asarray(k), cp.asarray(v)


def med_p90(samples):
    a = np.asarray(samples, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.min(a)), float(np.max(a))


def timed_cupy(fn, cp, warmup, iterations, repeats):
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
    return med_p90(samples), samples


def timed_torch(fn, torch, warmup, iterations, repeats):
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
    return med_p90(samples), samples


def cold_ms(fn, sync):
    t0 = time.perf_counter()
    fn()
    sync()
    return (time.perf_counter() - t0) * 1000.0


def probe_sdpa_backend(sdpa_fn, torch):
    """Return profiler-observed aten operators, keeping probe failure non-fatal."""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            sdpa_fn()
            torch.cuda.synchronize()
        wanted = []
        for event in prof.key_averages():
            key = str(event.key)
            lower = key.lower()
            if ("scaled_dot_product" in lower or "flash_attention" in lower
                    or "efficient_attention" in lower or "cudnn_attention" in lower):
                wanted.append(key)
        return {"operator_names": sorted(set(wanted)), "method": "torch.profiler", "status": "captured" if wanted else "no SDPA backend operator surfaced"}
    except Exception as exc:
        return {"operator_names": [], "method": "torch.profiler", "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/cutile-vs-pytorch-sdpa.json"))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--repeats-per-sample", type=int, default=5)
    args = p.parse_args()
    if args.warmup < 1 or args.iterations < 30 or args.repeats_per_sample < 1:
        p.error("warmup >= 1, iterations >= 30, repeats-per-sample >= 1 required")

    try:
        import cupy as cp
        import torch
        import torch.nn.functional as F
        import cuda.tile  # noqa: F401
    except Exception as exc:
        print(f"ERROR: required CuPy/cuTile/PyTorch import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: PyTorch reports CUDA unavailable", file=sys.stderr)
        return 2

    torch_device = torch.device("cuda:0")
    prop = torch.cuda.get_device_properties(torch_device)
    gpu_name = prop.name
    if cp.cuda.runtime.getDeviceCount() < 1:
        print("ERROR: CuPy reports no CUDA device", file=sys.stderr)
        return 2

    case_map = {c.name: c for c in MANDATORY_CASES}
    input_cache = {}
    sdpa_cold = {}
    backend_cache = {}
    tile_cold = {}
    rows = []
    raw_sessions = []
    session_orders = (
        ("cutile", "sdpa"),
        ("sdpa", "cutile"),
        ("cutile", "sdpa"),
    )

    for case_name, tq, tk in CONFIGS:
        case = case_map[case_name]
        if case.N % tq or case.N % tk:
            continue
        if case_name not in input_cache:
            q_np, k_np, v_np, q_cp, k_cp, v_cp = make_inputs(case, cp)
            # Zero-copy DLPack bridges let both libraries consume the exact same GPU data.
            q_t = torch.from_dlpack(q_cp)
            k_t = torch.from_dlpack(k_cp)
            v_t = torch.from_dlpack(v_cp)
            input_cache[case_name] = (q_np, k_np, v_np, q_cp, k_cp, v_cp, q_t, k_t, v_t)
        q_np, k_np, v_np, q_cp, k_cp, v_cp, q_t, k_t, v_t = input_cache[case_name]

        expected = causal_attention_fp32(q_np, k_np, v_np, output_dtype=np.float16)

        def sdpa_fn():
            return F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=None, dropout_p=0.0, is_causal=True)

        def cutile_fn():
            return cutile_launch_device(q_cp, k_cp, v_cp, tile_q=tq, tile_k=tk)

        if case_name not in sdpa_cold:
            sdpa_cold[case_name] = cold_ms(sdpa_fn, torch.cuda.synchronize)
            backend_cache[case_name] = probe_sdpa_backend(sdpa_fn, torch)
        tile_key = (case_name, tq, tk)
        if tile_key not in tile_cold:
            tile_cold[tile_key] = cold_ms(cutile_fn, cp.cuda.get_current_stream().synchronize)

        # Correctness is checked before the timed experiment is accepted.
        sdpa_out = sdpa_fn()
        torch.cuda.synchronize()
        sdpa_np = sdpa_out.detach().cpu().numpy()
        if sdpa_np.dtype != np.float16 or not np.isfinite(sdpa_np).all():
            raise AssertionError(f"{case_name} SDPA returned wrong dtype or non-finite values")
        sdpa_pass = passes_mixed_tolerance(sdpa_np, expected, abs_scale=ABS_TOL, rel_scale=REL_TOL)
        sdpa_max_abs = max_abs_error(sdpa_np, expected)
        sdpa_max_rel = max_rel_error(sdpa_np, expected)

        cutile_out = cutile_fn()
        cp.cuda.get_current_stream().synchronize()
        cutile_np = cp.asnumpy(cutile_out)
        if cutile_np.dtype != np.float16 or not np.isfinite(cutile_np).all():
            raise AssertionError(f"{case_name}/{tq}x{tk} cuTile returned wrong dtype or non-finite values")
        cutile_pass = passes_mixed_tolerance(cutile_np, expected, abs_scale=ABS_TOL, rel_scale=REL_TOL)
        cutile_max_abs = max_abs_error(cutile_np, expected)
        cutile_max_rel = max_rel_error(cutile_np, expected)
        cross_pass = passes_mixed_tolerance(cutile_np, sdpa_np, abs_scale=ABS_TOL, rel_scale=REL_TOL)
        if not (sdpa_pass and cutile_pass and cross_pass):
            raise AssertionError(
                f"{case_name}/{tq}x{tk} correctness failure: SDPA={sdpa_pass}, "
                f"cuTile={cutile_pass}, cross={cross_pass}, "
                f"abs(sdpa)={sdpa_max_abs}, abs(cuTile)={cutile_max_abs}"
            )

        output_bytes = case.B * case.H * case.N * case.Dv * np.dtype(np.float16).itemsize
        for session_index, order in enumerate(session_orders, 1):
            measurements = {}
            for backend_name in order:
                if backend_name == "cutile":
                    (med, p90, low, high), samples = timed_cupy(
                        cutile_fn, cp, args.warmup, args.iterations, args.repeats_per_sample
                    )
                else:
                    (med, p90, low, high), samples = timed_torch(
                        sdpa_fn, torch, args.warmup, args.iterations, args.repeats_per_sample
                    )
                measurements[backend_name] = {
                    "median_ms": med,
                    "p90_ms": p90,
                    "min_sample_ms": low,
                    "max_sample_ms": high,
                    "range_pct_of_median": (100.0 * (high - low) / med) if med else None,
                    "output_GiB_s": output_bytes / med / (2**30) * 1000.0 if med else None,
                    "samples_ms": samples,
                }
            row = {
                "session": session_index,
                "case": case_name,
                "B": case.B,
                "H": case.H,
                "N": case.N,
                "D": case.D,
                "tile_q": tq,
                "tile_k": tk,
                "measurement_order": list(order),
                "cutile_cold_first_call_ms": tile_cold[tile_key],
                "sdpa_cold_first_call_ms": sdpa_cold[case_name],
                "sdpa_backend_probe": backend_cache[case_name],
                "correctness": {
                    "sdpa_vs_numpy_pass": sdpa_pass,
                    "cutile_vs_numpy_pass": cutile_pass,
                    "cutile_vs_sdpa_pass": cross_pass,
                    "sdpa_max_abs_error": sdpa_max_abs,
                    "sdpa_max_rel_error": sdpa_max_rel,
                    "cutile_max_abs_error": cutile_max_abs,
                    "cutile_max_rel_error": cutile_max_rel,
                    "tolerance": f"abs(error) <= {ABS_TOL} + {REL_TOL}*abs(reference)",
                },
                "measurements": measurements,
                "sdpa_speedup_over_cutile": (
                    measurements["cutile"]["median_ms"] / measurements["sdpa"]["median_ms"]
                    if measurements["sdpa"]["median_ms"] else None
                ),
            }
            rows.append(row)
            raw_sessions.append(row)

        print(
            f"{case_name:24s} tile={tq}x{tk} "
            f"SDPA={statistics.median([r['measurements']['sdpa']['median_ms'] for r in rows if r['case']==case_name and r['tile_q']==tq and r['tile_k']==tk]):.4f}ms "
            f"cuTile={statistics.median([r['measurements']['cutile']['median_ms'] for r in rows if r['case']==case_name and r['tile_q']==tq and r['tile_k']==tk]):.4f}ms "
            f"PASS"
        )

    # Summarize the paired three-session measurements; retain raw samples above.
    grouped = {}
    for row in rows:
        key = (row["case"], row["tile_q"], row["tile_k"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (case_name, tq, tk), items in sorted(grouped.items()):
        sdpa_medians = [item["measurements"]["sdpa"]["median_ms"] for item in items]
        cutile_medians = [item["measurements"]["cutile"]["median_ms"] for item in items]
        sdpa_med = float(statistics.median(sdpa_medians))
        cutile_med = float(statistics.median(cutile_medians))
        summary.append({
            "case": case_name,
            "tile_q": tq,
            "tile_k": tk,
            "session_count": len(items),
            "sdpa_median_ms": sdpa_med,
            "sdpa_p90_ms": float(statistics.median([item["measurements"]["sdpa"]["p90_ms"] for item in items])),
            "cutile_median_ms": cutile_med,
            "cutile_p90_ms": float(statistics.median([item["measurements"]["cutile"]["p90_ms"] for item in items])),
            "sdpa_speedup_over_cutile": (cutile_med / sdpa_med) if sdpa_med else None,
            "sdpa_session_range_pct": (100.0 * (max(sdpa_medians)-min(sdpa_medians)) / sdpa_med) if sdpa_med else None,
            "cutile_session_range_pct": (100.0 * (max(cutile_medians)-min(cutile_medians)) / cutile_med) if cutile_med else None,
            "sdpa_backend": items[-1]["sdpa_backend_probe"],
            "all_correctness_pass": all(all(row["correctness"].get(k, False) for k in ("sdpa_vs_numpy_pass","cutile_vs_numpy_pass","cutile_vs_sdpa_pass")) for row in items),
        })

    gpu = cp.cuda.runtime.getDeviceProperties(0)
    name = gpu.get("name", b"")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    env = {
        "gpu": str(name),
        "compute_capability": [int(prop.major), int(prop.minor)],
        "driver_version": str(subprocess_check_driver()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cuda_tile": importlib.metadata.version("cuda-tile"),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_sdpa_api": "torch.nn.functional.scaled_dot_product_attention",
        "sdpa_mask": "is_causal=True, no explicit dense mask",
        "input_dtype": "float16",
        "output_dtype": "float16",
        "warmup": args.warmup,
        "iterations_per_session": args.iterations,
        "launches_per_event_sample": args.repeats_per_sample,
        "timed_launches_per_backend_per_session": args.iterations * args.repeats_per_sample,
        "session_count": len(session_orders),
        "session_order": [list(x) for x in session_orders],
        "torch_default_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "torch_default_mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "torch_default_cudnn_sdp_enabled": bool(getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: False)()),
        "baseline_note": "Existing unfused FP32 CuPy baseline is retained in the project benchmark. This experiment adds vendor/framework SDPA and directly compares it with the fused cuTile kernel using identical GPU-resident FP16 Q/K/V.",
    }
    payload = {
        "title": "Actual cuTile vs CUDA PyTorch SDPA",
        "environment": env,
        "summary": summary,
        "raw_sessions": raw_sessions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = list(summary[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({**row, "sdpa_backend": json.dumps(row["sdpa_backend"])})
    print(f"SDPA_REPORT_JSON={args.output}")
    print(f"SDPA_REPORT_CSV={csv_path}")
    print(f"PYTORCH={torch.__version__} CUDA_BUILD={torch.version.cuda} GPU={name}")
    return 0


def subprocess_check_driver():
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
