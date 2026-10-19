#!/usr/bin/env python3
"""Inference-only PyTorch block integration for the cuTile causal-attention kernel.

The baseline and cuTile models share the same projection weights. This verifies
that DLPack tensor exchange can place the existing fused kernel inside an
ordinary torch.nn.Module without changing the kernel contract. It does not
provide native cuTile backward propagation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ModelAttention:
    """Delayed factory keeps Torch imports out of CPU-only module import."""

    def __new__(cls, heads=8, head_dim=64, cutile=False, tile_q=64, tile_k=64):
        import cupy as cp
        import torch
        from torch import nn
        import torch.nn.functional as F
        from cutile_attention import launch_device

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.heads = heads
                self.head_dim = head_dim
                self.model_dim = heads * head_dim
                self.use_cutile = cutile
                self.tile_q = tile_q
                self.tile_k = tile_k
                self.q_proj = nn.Linear(self.model_dim, self.model_dim)
                self.k_proj = nn.Linear(self.model_dim, self.model_dim)
                self.v_proj = nn.Linear(self.model_dim, self.model_dim)
                self.out_proj = nn.Linear(self.model_dim, self.model_dim)

            def _split(self, x):
                batch, seq, _ = x.shape
                return x.view(batch, seq, self.heads, self.head_dim).transpose(1, 2).contiguous()

            def forward(self, x):
                q = self._split(self.q_proj(x))
                k = self._split(self.k_proj(x))
                v = self._split(self.v_proj(x))
                if not self.use_cutile:
                    attended = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                else:
                    # DLPack shares the CUDA storage; no host copy of Q/K/V or O.
                    q_cp = cp.from_dlpack(q)
                    k_cp = cp.from_dlpack(k)
                    v_cp = cp.from_dlpack(v)
                    out_cp = launch_device(
                        q_cp, k_cp, v_cp, tile_q=self.tile_q, tile_k=self.tile_k
                    )
                    attended = torch.from_dlpack(out_cp)
                attended = attended.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.model_dim)
                return self.out_proj(attended)

        return Module()


def benchmark(fn, torch, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / repeats)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "sample_count": len(values),
        "repeats_per_sample": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats-per-sample", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "framework-attention" / "model-integration.json")
    args = parser.parse_args()
    if args.sequence % 64:
        parser.error("sequence length must be divisible by the 64x64 cuTile configuration")
    if args.iterations < 30:
        parser.error("at least 30 event samples are required")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this integration experiment")
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False

    baseline = ModelAttention(args.heads, args.head_dim, cutile=False).cuda().half().eval()
    fused = ModelAttention(args.heads, args.head_dim, cutile=True).cuda().half().eval()
    fused.load_state_dict(baseline.state_dict())
    x = torch.randn((args.batch, args.sequence, args.heads * args.head_dim), device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        t0 = time.perf_counter()
        ref = baseline(x)
        torch.cuda.synchronize()
        baseline_first_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        actual = fused(x)
        torch.cuda.synchronize()
        cutile_first_ms = (time.perf_counter() - t0) * 1000.0

        if actual.shape != ref.shape or actual.dtype != torch.float16 or not torch.isfinite(actual).all():
            raise AssertionError("model output contract failed")
        diff = (actual.float() - ref.float()).abs()
        allowed = 2e-2 + 2e-2 * ref.float().abs()
        violations = int((diff > allowed).sum().item())
        if violations:
            raise AssertionError(f"model integration differs from SDPA baseline at {violations} elements")

        base_fn = lambda: baseline(x)
        fused_fn = lambda: fused(x)
        base_timing = benchmark(base_fn, torch, args.warmup, args.iterations, args.repeats_per_sample)
        fused_timing = benchmark(fused_fn, torch, args.warmup, args.iterations, args.repeats_per_sample)

    result = {
        "status": "PASS",
        "mode": "inference-only; native cuTile backward is not implemented by this wrapper",
        "seed": 20261019,
        "shape": {
            "batch": args.batch,
            "sequence": args.sequence,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "model_dim": args.heads * args.head_dim,
        },
        "dtype": "FP16",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "baseline_backend": "torch.nn.functional.scaled_dot_product_attention; automatic CUDA backend selection",
        "cuTile_exchange": "DLPack zero-copy tensor exchange",
        "first_call_wall_ms": {
            "sdpa_model": baseline_first_ms,
            "cuTile_model": cutile_first_ms,
        },
        "steady_state": {
            "sdpa_model": base_timing,
            "cuTile_model": fused_timing,
            "speedup_sdpa_over_cutile": base_timing["median_ms"] / fused_timing["median_ms"],
        },
        "numerical_comparison": {
            "max_abs_error": float(diff.max().item()),
            "max_relative_error_abs_ref_ge_1e_2": float(
                (diff[ref.float().abs() >= 1e-2] / ref.float().abs()[ref.float().abs() >= 1e-2]).max().item()
            ),
            "tolerance_violations": violations,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"MODEL_INTEGRATION=PASS shape={result['shape']} "
        f"SDPA={base_timing['median_ms']:.6f}ms "
        f"cuTile={fused_timing['median_ms']:.6f}ms "
        f"speedup={result['steady_state']['speedup_sdpa_over_cutile']:.3f}x"
    )
    print(f"MODEL_RESULT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
