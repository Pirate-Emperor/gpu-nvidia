#!/usr/bin/env python3
"""Compile preserved shared-tile kernels with NVCC/PTXAS resource reporting."""

from __future__ import annotations

import ast
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_FILES = {
    "v1": ROOT / "src" / "variants" / "shared_tile_v1.py",
    "v2": ROOT / "src" / "variants" / "shared_tile_v2.py",
    "v3": ROOT / "src" / "variants" / "shared_tile_v3.py",
    "current": ROOT / "src" / "shared_tile_cuda_attention.py",
    "v4": ROOT / "src" / "variants" / "shared_tile_v4.py",
}


def extract_source(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_kernel":
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    if stmt.targets[0].id == "source" and isinstance(stmt.value, ast.Constant):
                        if isinstance(stmt.value.value, str):
                            return stmt.value.value
    raise RuntimeError(f"could not extract kernel source from {path}")


def main() -> int:
    out = {}
    with tempfile.TemporaryDirectory(prefix="tile-ptxas-") as td:
        tdir = Path(td)
        for name, path in KERNEL_FILES.items():
            src = tdir / f"{name}.cu"
            cubin = tdir / f"{name}.cubin"
            src.write_text(extract_source(path))
            cmd = [
                "nvcc", "-O3", "-std=c++14", "-arch=sm_80",
                "-Xptxas=-v", "-cubin", str(src), "-o", str(cubin),
            ]
            p = subprocess.run(cmd, text=True, capture_output=True)
            out[name] = {
                "returncode": p.returncode,
                "stdout": p.stdout,
                "stderr": p.stderr,
                "cubin_bytes": cubin.stat().st_size if p.returncode == 0 and cubin.exists() else None,
            }
            if p.returncode != 0:
                print(f"FAILED {name}\n{p.stderr}", flush=True)
                return p.returncode

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
