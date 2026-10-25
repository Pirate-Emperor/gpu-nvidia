#!/usr/bin/env python3
"""End-to-end model-block comparison: cuTile forward bridge vs PyTorch SDPA."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from torch_integration import make_attention_module


def time_model(model, x, *, warmup, iterations, repeats):
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    samples = []
    with torch.inference_mode():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                model(x)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) / repeats)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_sample_ms": float(values.min()),
        "max_sample_ms": float(values.max()),
        "samples_ms": [float(x) for x in values],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/cutile-vs-sdpa-model.json"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sessions", type=int, default=3)
    args = parser.parse_args()
    if args.sequence % 64:
        parser.error("sequence length must be divisible by 64 for this initial model-block experiment")
    if args.embed_dim % args.heads or args.embed_dim // args.heads not in (32, 64, 128):
        parser.error("per-head dimension must be 32, 64 or 128")
    if args.iterations < 30 or args.warmup < 1 or args.repeats < 1 or args.sessions < 3:
        parser.error("require warmup >= 1, iterations >= 30, repeats >= 1, sessions >= 3")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports CUDA unavailable")

    CuTileBlock, SDPABlock = make_attention_module(torch)
    torch.manual_seed(20261019)
    cutile_model = CuTileBlock(
        args.embed_dim, args.heads, tile_q=64, tile_k=64
    ).cuda().half().eval()
    sdpa_model = SDPABlock(args.embed_dim, args.heads).cuda().half().eval()
    sdpa_model.load_state_dict(cutile_model.state_dict())
    x = torch.randn(
        (args.batch, args.sequence, args.embed_dim),
        device="cuda", dtype=torch.float16,
    )

    with torch.inference_mode():
        y_cutile = cutile_model(x)
        y_sdpa = sdpa_model(x)
        torch.cuda.synchronize()
    max_abs = float((y_cutile.float() - y_sdpa.float()).abs().max().item())
    rmse = float(torch.mean((y_cutile.float() - y_sdpa.float()) ** 2).sqrt().item())
    tol = 2e-2 + 2e-2 * y_sdpa.float().abs()
    tolerance_pass = bool(torch.all((y_cutile.float() - y_sdpa.float()).abs() <= tol).item())
    if y_cutile.shape != y_sdpa.shape or not torch.isfinite(y_cutile).all().item():
        raise AssertionError("model outputs have shape/non-finite failure")
    if not tolerance_pass:
        raise AssertionError(
            f"cuTile model-block output does not meet mixed tolerance; max_abs={max_abs}"
        )

    session_orders = (("cutile", "sdpa"), ("sdpa", "cutile"), ("cutile", "sdpa"))
    raw = []
    for session_index, order in enumerate(session_orders[:args.sessions], 1):
        measurements = {}
        for backend in order:
            model = cutile_model if backend == "cutile" else sdpa_model
            measurements[backend] = time_model(
                model, x, warmup=args.warmup,
                iterations=args.iterations, repeats=args.repeats,
            )
        raw.append({
            "session": session_index,
            "order": list(order),
            "measurements": measurements,
        })

    cutile_medians = [item["measurements"]["cutile"]["median_ms"] for item in raw]
    sdpa_medians = [item["measurements"]["sdpa"]["median_ms"] for item in raw]
    cutile_median = float(statistics.median(cutile_medians))
    sdpa_median = float(statistics.median(sdpa_medians))
    payload = {
        "title": "End-to-end causal attention model block: cuTile vs PyTorch SDPA",
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cuda_tile": importlib.metadata.version("cuda-tile"),
            "batch": args.batch,
            "sequence": args.sequence,
            "embed_dim": args.embed_dim,
            "heads": args.heads,
            "dtype": "FP16",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats_per_event_sample": args.repeats,
            "sessions": args.sessions,
            "state_dict_shared": True,
        },
        "correctness": {
            "output_shape_equal": list(y_cutile.shape) == list(y_sdpa.shape),
            "all_finite": bool(torch.isfinite(y_cutile).all().item()),
            "max_abs_error": max_abs,
            "rmse": rmse,
            "mixed_tolerance_pass": tolerance_pass,
            "tolerance": "abs(actual-reference) <= 0.02 + 0.02*abs(reference)",
        },
        "summary": {
            "cutile_median_ms": cutile_median,
            "cutile_p90_ms": float(statistics.median(item["measurements"]["cutile"]["p90_ms"] for item in raw)),
            "sdpa_median_ms": sdpa_median,
            "sdpa_p90_ms": float(statistics.median(item["measurements"]["sdpa"]["p90_ms"] for item in raw)),
            "speedup_sdpa_over_cutile": cutile_median / sdpa_median if sdpa_median else None,
            "cutile_session_range_pct": 100.0 * (max(cutile_medians) - min(cutile_medians)) / cutile_median,
            "sdpa_session_range_pct": 100.0 * (max(sdpa_medians) - min(sdpa_medians)) / sdpa_median,
        },
        "raw_sessions": raw,
        "method": "Same QKV projections, same state_dict, same FP16 input and output contract; includes projection and output projection; cold model/JIT timing is excluded and not included in the CUDA-event steady-state samples.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"MODEL_REPORT_JSON={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
