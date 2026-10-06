#!/usr/bin/env python3
"""Exhaustive read-only GPU/node inventory for the project's self-hosted A100 runner.

Second PR execution revision: collect actual runner state before any admin change."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess


def run(argv, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)
        return f"exit={p.returncode}\n{(p.stdout or p.stderr).strip()}"
    except Exception as exc:
        return f"<unavailable: {exc}>"


def cmd(name):
    return shutil.which(name) or "<not found>"


def section(title, body):
    print("\n" + "=" * 14 + f" {title} " + "=" * 14)
    print(body)


def env(name):
    return os.environ.get(name, "<unset>")


def main():
    print("GPU RUNNER EXHAUSTIVE AUDIT")
    print("READ-ONLY: no driver/package/device mutation")

    section("RUNNER IDENTITY", "\n".join([
        f"hostname: {platform.node()}",
        f"os: {platform.platform()}",
        f"kernel: {platform.release()}",
        f"arch: {platform.machine()}",
        f"python: {platform.python_version()}",
        f"uid/euid: {os.getuid()}/{os.geteuid()}",
        f"github_actions: {env('GITHUB_ACTIONS')}",
        f"runner_name: {env('RUNNER_NAME')}",
        f"workspace: {env('GITHUB_WORKSPACE')}",
        f"cuda_visible_devices: {env('CUDA_VISIBLE_DEVICES')}",
        f"cuda_home: {env('CUDA_HOME')}",
        f"cuda_path: {env('CUDA_PATH')}",
    ]))

    section("GPU SUMMARY", run([
        "nvidia-smi",
        "--query-gpu=index,gpu_uuid,name,pci.bus_id,vbios_version,driver_version,compute_cap,"
        "memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
        "pstate,persistence_mode,compute_mode,mig.mode.current,ecc.mode.current",
        "--format=csv",
    ]))

    section("GPU LIST", run(["nvidia-smi", "-L"]))
    section("GPU FULL QUERY", run(["nvidia-smi", "-q"], timeout=30))
    section("TOPOLOGY MATRIX", run(["nvidia-smi", "topo", "-m"]))
    section("MIG GPU INSTANCES", run(["nvidia-smi", "mig", "-lgi"]))
    section("MIG COMPUTE INSTANCES", run(["nvidia-smi", "mig", "-lci"]))

    section("NVLINK STATUS", run(["nvidia-smi", "nvlink", "-s"]))
    section("NVLINK PEERS", run(["nvidia-smi", "nvlink", "-p"]))

    section("PCI DEVICES", run(["lspci", "-nn", "-d", "10de:"]))
    section("PCI NVIDIA DETAIL", run(["lspci", "-vv", "-d", "10de:"], timeout=30))

    section("CUDA TOOLCHAIN", "\n".join([
        "nvcc path: " + cmd("nvcc"),
        run(["nvcc", "--version"]),
        "ptxas path: " + cmd("ptxas"),
        "ptxas version: " + (run(["ptxas", "--version"]) if shutil.which("ptxas") else "<not found>"),
        "nvdisasm path: " + cmd("nvdisasm"),
        "cuobjdump path: " + cmd("cuobjdump"),
    ]))

    section("NSIGHT", "\n".join([
        "ncu path: " + cmd("ncu"),
        run(["ncu", "--version"], timeout=20) if shutil.which("ncu") else "<not found>",
        "nsys path: " + cmd("nsys"),
        run(["nsys", "--version"], timeout=20) if shutil.which("nsys") else "<not found>",
    ]))

    section("FABRIC MANAGER", "\n".join([
        "nv-fabricmanager path: " + cmd("nv-fabricmanager"),
        "service active: " + run(["systemctl", "is-active", "nvidia-fabricmanager"]),
        "service status:\n" + run(["systemctl", "status", "--no-pager", "nvidia-fabricmanager"], timeout=20),
    ]))

    section("KERNEL NVIDIA MODULES", run([
        "bash", "-lc", "lsmod | grep -E 'nvidia|nvlink|nvswitch' || true"
    ]))
    section("NVIDIA MODULE VERSION", run([
        "bash", "-lc",
        "modinfo nvidia 2>/dev/null | egrep '^(version|filename):' || true"
    ]))

    section("DEVICE NODE ACCESS", run([
        "bash", "-lc",
        "id; ls -l /dev/nvidia* 2>/dev/null || true; "
        "stat -c '%n %a %U:%G' /dev/nvidia* 2>/dev/null || true"
    ]))

    section("LIVE UTILIZATION", run([
        "nvidia-smi", "dmon", "-s", "pucm", "-c", "3"
    ], timeout=20))

    section("THERMAL/POWER/CLOCKS", run([
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,temperature.gpu.tlimit,power.draw,power.limit,"
        "clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,"
        "ecc.errors.uncorrected.volatile.total",
        "--format=csv",
    ]))

    section("CUDA LIBRARIES / ENV", "\n".join([
        run(["bash", "-lc",
             "ldconfig -p 2>/dev/null | grep -E 'libcuda|libcudart|libnvrtc|libnvvm|libnvidia-ml' || true"]),
        "LD_LIBRARY_PATH: " + env("LD_LIBRARY_PATH"),
        "PATH: " + env("PATH"),
    ]))

    section("HOST COMPILER", "\n".join([
        "gcc:\n" + run(["gcc", "--version"]),
        "g++:\n" + run(["g++", "--version"]),
        "glibc:\n" + run(["ldd", "--version"]),
    ]))

    section("DCGM", "\n".join([
        "dcgmi path: " + cmd("dcgmi"),
        run(["dcgmi", "discovery", "-l"]) if shutil.which("dcgmi") else "<not found>",
        "nv-hostengine path: " + cmd("nv-hostengine"),
    ]))

    section("PERFORMANCE COUNTER / PROFILING ACCESS", "\n".join([
        "ncu availability: " + ("yes" if shutil.which("ncu") else "no"),
        "device node permissions checked above",
        "CUDA_VISIBLE_DEVICES: " + env("CUDA_VISIBLE_DEVICES"),
    ]))

    print("\n=== END AUDIT ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
