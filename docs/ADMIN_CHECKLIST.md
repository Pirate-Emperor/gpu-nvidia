# Admin / Cluster Prerequisite Checklist

Use this checklist before declaring the A100/cuTile environment ready.

## Required

1. A100/SM80 GPU exposed to the runner.
2. NVIDIA Data Center Driver R580 or later.
3. For detailed cuTile Nsight Compute statistics, use R580.126.09 or later. NVIDIA's current documented R580 release is 580.178.04.
4. Python 3.10-3.14 available; the project CI uses Python 3.12.
5. A CUDA 13.1+ compiler/toolchain path for cuTile: either a system CUDA Toolkit 13.1+ installation or the project virtual environment can provide TileIR compiler components through `cuda-tile[tileiras]`.
6. Current Nsight Compute CLI (`ncu`) installed on the target node for the required profiling runs.

## Required on HGX/NVSwitch systems

- Driver and NVIDIA Fabric Manager should be installed as a compatible/matching release.
- The node should be drained/rebooted according to the cluster's normal driver-maintenance procedure.

## Profiling access

- The GitHub Actions runner user is non-root.
- Non-admin access to NVIDIA GPU performance counters must therefore be enabled for the runner, otherwise Nsight Compute can fail with `ERR_NVGPUCTRPERM`.
- During profiling, concurrent tools such as DCGM or another profiler may need coordination because they can reserve the performance-monitoring resource.

## Benchmarking

- Prefer one exclusive physical A100 for the benchmark/profiling job.
- Prefer MIG disabled for the benchmark; a MIG slice changes the available resources and makes the reported performance non-comparable to a full 40GB A100.
- Do not force clocks unless the cluster policy permits it. Instead, record GPU clocks, power limit, temperature, and utilization for every benchmark session.
- Record the GPU UUID/topology so results can be tied to the exact device.

## Important: what does NOT necessarily need a host-wide upgrade

A full system CUDA Toolkit upgrade is not strictly required merely because the existing host `nvcc` is old. NVIDIA's current cuTile quickstart permits the required TileIR compiler pieces (`nvidia-cuda-tileiras`, `nvidia-cuda-nvcc`, and `nvidia-nvvm`) to be installed inside the project's Python environment when system CUDA 13.1+ is unavailable. The package versions must match by major.minor.

Therefore the preferred low-impact admin change is:

`R580.126.09+ driver (ideally the current R580 release) + matching Fabric Manager where applicable + current Nsight Compute + non-admin profiling permission`.

Then the project can supply its isolated CUDA/cuTile compiler components in the virtual environment.

## Verification

Run the read-only preflight:

    python3 scripts/cutile_sanity_check.py

Then verify the cuTile environment with:

    nvidia-smi
    nvidia-smi -L
    nvidia-smi topo -m
    nvcc --version
    ncu --version
    systemctl is-active nvidia-fabricmanager   # on systems using Fabric Manager

Do not treat a successful `nvidia-smi` check alone as sufficient; the driver, compiler, profiler, profiling permissions, and GPU allocation policy all matter.

## Confirmed current node findings

The self-hosted runner is on an 8-GPU HGX/NVSwitch system:
- 8 physical A100-SXM4-40GB GPUs are visible to DCGM.
- 6 NVSwitches are present.
- The runner exposes GPUs 0 and 1 through CUDA_VISIBLE_DEVICES=0,1.
- The selected GPUs are full GPUs, not MIG instances; MIG is disabled.
- Both selected GPUs were idle at the audit snapshot, with 40960 MiB installed and only 1 MiB used.
- Both selected GPUs reported PCIe Gen4 x16.
- NVLink status showed 12 links at 25 GB/s on each selected A100.
- Fabric Manager is installed and active.
- System CUDA Toolkit is 12.4.131.
- System GCC is 11.4.0 and glibc is 2.35.
- Nsight Compute 2024.1.1.0 and Nsight Systems 2023.4.4.54 are installed.
- Basic CUDA execution succeeds on the runner.
- CuPy RawKernel JIT currently fails because NVRTC cannot find standard headers such as float.h and math.h. This is being fixed with the matching CUDA 12.4 runtime-header package in the project environment rather than requiring a host-wide CUDA upgrade.
