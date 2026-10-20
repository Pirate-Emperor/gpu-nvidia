#!/usr/bin/env python3
"""Isolated PyTorch CUDA GEMM probe for comparing installed CUDA wheel builds.

Each operation runs in a separate child process so a failing cuBLAS handle or
CUDA context cannot poison the remaining cases. The parent always writes JSON
and exits successfully; workflow policy can then decide whether to gate on it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = ("mm-fp32", "bmm-batch1-fp32", "bmm-batch2-fp32", "linear-fp16")


def single_case(name: str) -> int:
    import torch

    result = {
        "case": name,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        torch.cuda.init()
        torch.manual_seed(90817)
        if name == "mm-fp32":
            a = torch.randn((64, 32), device="cuda", dtype=torch.float32)
            b = torch.randn((32, 64), device="cuda", dtype=torch.float32)
            out = torch.mm(a, b)
        elif name == "bmm-batch1-fp32":
            a = torch.randn((1, 64, 32), device="cuda", dtype=torch.float32)
            b = torch.randn((1, 64, 32), device="cuda", dtype=torch.float32)
            out = torch.bmm(a, b.transpose(1, 2).contiguous())
        elif name == "bmm-batch2-fp32":
            a = torch.randn((2, 64, 32), device="cuda", dtype=torch.float32)
            b = torch.randn((2, 64, 32), device="cuda", dtype=torch.float32)
            out = torch.bmm(a, b.transpose(1, 2).contiguous())
        elif name == "linear-fp16":
            x = torch.randn((1, 64, 128), device="cuda", dtype=torch.float16)
            weight = torch.randn((384, 128), device="cuda", dtype=torch.float16)
            out = torch.nn.functional.linear(x, weight)
        else:
            raise ValueError(f"unknown case: {name}")
        torch.cuda.synchronize()
        if not torch.isfinite(out).all().item():
            raise AssertionError("operation produced a non-finite output")
        result.update({
            "passed": True,
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "gpu": torch.cuda.get_device_name(0),
        })
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        result.update({
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        print(json.dumps(result, sort_keys=True))
        return 1


def parent(output: Path) -> int:
    rows = []
    for name in CASES:
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--single", name],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
                env=os.environ.copy(),
            )
            stdout = proc.stdout.strip().splitlines()
            row = json.loads(stdout[-1]) if stdout else {
                "case": name, "passed": False, "error": proc.stderr.strip() or "no child output"
            }
            row["returncode"] = proc.returncode
            if proc.stderr.strip():
                row["stderr_tail"] = proc.stderr.strip()[-2000:]
        except Exception as exc:
            row = {"case": name, "passed": False, "error_type": type(exc).__name__, "error": str(exc)}
        row["wall_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "purpose": "isolated PyTorch CUDA GEMM/cuBLAS diagnostic",
        "python": sys.version.split()[0],
        "cases": rows,
        "all_passed": all(row.get("passed") for row in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"TORCH_GEMM_DIAGNOSTIC={json.dumps({'all_passed': payload['all_passed'], 'output': str(output)})}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", choices=CASES)
    parser.add_argument("--output", type=Path, default=Path("results/torch-cu132-gemm-diagnostic.json"))
    args = parser.parse_args()
    return single_case(args.single) if args.single else parent(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
