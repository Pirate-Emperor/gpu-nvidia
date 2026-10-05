#!/usr/bin/env python3
"""Read-only environment audit for the A100 cuTile project."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class Check:
    name: str
    value: str
    status: str


def run_cmd(argv: list[str]) -> Optional[str]:
    try:
        p = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (p.stdout or p.stderr).strip()
    return out or None


def module_version(name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None) or "importable"


def main() -> int:
    checks: list[Check] = []

    checks.append(Check("Python", platform.python_version(), "OK"))
    checks.append(Check("OS", platform.platform(), "INFO"))
    checks.append(Check("GPU", run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]) or "not found", "INFO"))
    checks.append(Check("Driver", run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or "not found", "INFO"))
    checks.append(Check("Compute Capability", run_cmd(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]) or "not reported", "INFO"))
    checks.append(Check("CUDA compiler", run_cmd(["nvcc", "--version"]) or "not found", "INFO"))
    checks.append(Check("PyTorch", module_version("torch") or "not importable", "INFO"))
    checks.append(Check("cuTile", module_version("cutile") or "not importable", "INFO"))
    checks.append(Check("NumPy", module_version("numpy") or "not importable", "INFO"))
    checks.append(Check("Nsight Compute", run_cmd(["ncu", "--version"]) or "not found", "INFO"))
    checks.append(Check("core_config", module_version("core_config") or "not importable", "INFO"))
    checks.append(Check("Git", shutil.which("git") or "not found", "INFO"))

    for check in checks:
        print(f"[{check.status}] {check.name}: {check.value}")

    # A GPU audit must not mutate the machine and should remain usable even
    # when optional project packages have not yet been installed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
