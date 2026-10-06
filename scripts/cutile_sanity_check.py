#!/usr/bin/env python3
"""Read-only preflight for the cuTile/A100 project.

This script never installs packages, changes driver state, or modifies the host.
It is intended to expose hidden cluster prerequisites before a cuTile run.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def run(argv):
    try:
        p = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<unavailable: {exc}>"
    return (p.stdout or p.stderr).strip()


def first_line(value):
    return value.splitlines()[0].strip() if value else "<none>"


def major_minor(version_text):
    match = re.search(r"release\s+(\d+)\.(\d+)", version_text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def driver_tuple(version_text):
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text)
    return tuple(map(int, match.groups())) if match else None


def main():
    print("=== Host ===")
    print("hostname:", platform.node())
    print("os:", platform.platform())
    print("kernel:", platform.release())
    print("arch:", platform.machine())
    print("glibc:", first_line(run(["ldd", "--version"])))
    print("python:", platform.python_version())
    print("gcc:", first_line(run(["gcc", "--version"])))
    print("uid/euid:", os.getuid(), os.geteuid())

    print("\n=== GPU ===")
    gpu_query = run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap,mig.mode.current,compute_mode,persistence_mode",
        "--format=csv",
    ])
    print(gpu_query)

    driver_text = run([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
    ])
    driver = driver_tuple(driver_text)
    if driver:
        ok_runtime = driver[0] >= 580
        ok_profile = driver >= (580, 126, 9)
        print("cuTile runtime driver gate (R580+):", "PASS" if ok_runtime else "FAIL")
        print("cuTile detailed profiling gate (R580.126.09+):", "PASS" if ok_profile else "FAIL")

    print("gpu topology:")
    print(run(["nvidia-smi", "-L"]))
    print("topology matrix:")
    print(run(["nvidia-smi", "topo", "-m"]))
    print("mig config:")
    print(run(["nvidia-smi", "-q", "-d", "MIG"]))
    print("gpu health:")
    print(run([
        "nvidia-smi",
        "--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,ecc.errors.uncorrected.volatile.total",
        "--format=csv",
    ]))

    print("\n=== CUDA toolchain ===")
    nvcc = run(["nvcc", "--version"])
    print(nvcc)
    toolkit = major_minor(nvcc)
    if toolkit:
        print(
            "system CUDA Toolkit >= 13.1:",
            "PASS" if toolkit >= (13, 1) else "INFO (can use Python tileiras package instead)",
        )
    else:
        print("system CUDA Toolkit: not detected; Python tileiras package can supply compiler pieces")

    print("\n=== Nsight Compute ===")
    print("ncu path:", shutil.which("ncu") or "<not found>")
    print(first_line(run(["ncu", "--version"])) if shutil.which("ncu") else "<not found>")

    print("\n=== Fabric Manager / NVSwitch ===")
    print("nv-fabricmanager path:", shutil.which("nv-fabricmanager") or "<not found>")
    if shutil.which("nv-fabricmanager"):
        print("nv-fabricmanager version:", first_line(run(["nv-fabricmanager", "-v"])))
    print("fabricmanager service:",
          first_line(run(["systemctl", "is-active", "nvidia-fabricmanager"])))

    print("\n=== Environment ===")
    for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PATH"):
        value = os.environ.get(name)
        print(f"{name}:", value if value else "<unset>")

    print("\n=== Project policy ===")
    print("Required runtime: A100/SM80 + R580+ + Python 3.10-3.14")
    print("Preferred profiling runtime: R580.126.09+")
    print("Preferred toolchain path: system CUDA 13.1+ OR matching tileiras/nvcc/nvvm packages in the venv")
    print("Profiling note: non-admin GPU performance-counter access must be permitted for an unprivileged runner")
    print("Profiling note: concurrent DCGM/performance-counter users may need coordination")
    print("Benchmark note: prefer an exclusive GPU allocation and stable clock/power policy")
    print("Benchmark note: full A100 40GB is preferred; record whether MIG is enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
