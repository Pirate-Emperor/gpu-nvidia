# GPU Environment Contract

## User's local GPU inventory — 2026-10-18

The user supplied the following command output in this conversation:

- GPU: NVIDIA GeForce RTX 4060, compute capability 8.9, PCI bus 00000000:01:00.0
- Driver: 616.92; NVIDIA-SMI reports CUDA UMD 13.4
- VRAM: 8,188 MiB total, 7,925 MiB free at capture
- OS: Windows 10 Pro, WindowsVersion 2009, build 26300, x64; driver model WDDM
- `nvcc`: not found
- Python: not available from `python`; Windows Microsoft Store alias responded
- WSL2: not installed
- Docker: not installed

**Assessment:** the GPU compute capability, driver, and Windows x86_64 OS meet the current [NVIDIA cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) prerequisites on paper (CC 8.x–12.x, R580+, Windows x86_64, Python 3.10–3.14 once installed). Driver 616.92 also exceeds the Quickstart's Windows R582.16+ threshold for detailed cuTile profiles. The system CUDA Toolkit is not present via `nvcc`, but the Quickstart allows compiler dependencies through the `cuda-tile[tileiras]` Python extra. The first path is therefore native Windows, with CPython 3.12 installed per-user and an isolated project venv. WSL2 and Docker are not necessary for the first smoke test.

There is a reported Windows cuTile/CuPy NVRTC builtins DLL failure in [NVIDIA issue #57](https://github.com/NVIDIA/cutile-python/issues/57), so compatibility remains conditional until `scripts/cutile_smoke.py` compiles, launches, synchronizes and verifies a real cuTile kernel. The project uses `cupy-cuda13x[ctk]` to let CuPy install matching CUDA component wheels inside the venv. No host driver/toolkit changes or GitHub runner registration are authorized by this inventory.

Follow [docs/LOCAL_GPU_WINDOWS.md](LOCAL_GPU_WINDOWS.md) for the exact setup and smoke-test procedure.

## Host versus user-space

The NVIDIA kernel-mode driver is a host property. CUDA Toolkit components and some compatibility libraries can be installed in user space, but a host driver upgrade is a system-level operation.

Current node observed by the project runner:

- GPU: NVIDIA A100-SXM4-40GB
- Compute capability: 8.0
- Driver: R550.90.07
- Current cuTile production gate: R580+

## Upgrade path

The clean path is:

1. Cluster administrator upgrades/maintains the node driver to a supported branch.
2. Reboot/maintenance is performed according to cluster policy.
3. The project runner re-checks nvidia-smi.
4. The cuTile environment is installed in an isolated Python environment.
5. The full correctness suite runs.
6. Benchmark/profiler workflows run.

The repository does not attempt a host driver mutation automatically.

## Actual-cuTile routing decision — 2026-10-18

Do not continue trying to force actual cuTile onto the existing A100 runner at R550.90.07. It remains useful for ordinary-CUDA surrogate tests but does not meet NVIDIA's documented R580+ cuTile Python driver prerequisite. Move actual-cuTile bring-up to the user's local GPU after collecting real inventory, or to another explicitly authorized supported GPU environment. Do not assume a local model or change a system driver without owner/admin action.

NVIDIA's [current cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) lists Linux x86_64, Linux aarch64 and Windows x86_64, compute capability 8.x–12.x, driver R580+, and Python 3.10–3.14. The optional `[tileiras]` installation supplies compiler components inside a Python environment; `nvidia-cuda-tileiras`, `nvidia-cuda-nvcc` and `nvidia-nvvm` must match in major.minor. A system CUDA Toolkit 13.1+ is an alternative, not automatically a global installation requirement. Detailed cuTile profiling requires R580.126.09+ on Linux or R582.16+ on Windows.

Choose Windows-native versus Linux/WSL2 only after gathering GPU model/compute capability, driver, OS/build or distro/kernel, WSL2 status, Python, Docker availability, and `nvcc --version`. The CUDA level shown by `nvidia-smi` is not proof NVCC/the Toolkit is installed. Do not register a runner until its OS-enforced isolation, mounts, privileges and workflow trigger policy have been verified.

## Forward compatibility

NVIDIA documents CUDA forward compatibility for supported data-center GPUs through cuda-compat-<major>-<minor> packages. The compatibility package supplies user-space driver components, including libcuda and, for newer CUDA releases, TileIR-related components.

The project tested cuda-compat-13-4 on the current node without root. The package downloaded and extracted successfully, but loading the compatibility libraries produced CUDA error 803 (system driver mismatch). Therefore that exact configuration is not currently a validated solution.

Official NVIDIA references:

- CUDA Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/
- Forward Compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
- cuTile Python Quickstart: https://docs.nvidia.com/cuda/cutile-python/quickstart.html

## Read-only access audit

scripts/driver_access_check.py reports:

- current UID/EUID;
- whether the process is root;
- whether sudo exists;
- whether passwordless sudo is available;
- package-manager availability;
- nvidia-smi/nvcc discovery.

It performs no installation, upgrade, driver unload, or reboot operation.


## Preflight checklist

Before a cuTile execution or profiling run, the project records:

- GPU model, compute capability, memory size, MIG state, compute mode, and persistence mode.
- NVIDIA driver version. Runtime requires R580+; detailed cuTile profiling requires R580.126.09+.
- System CUDA Toolkit version if present. NVIDIA's current cuTile quickstart supports a system CUDA Toolkit 13.1+ path; otherwise the project can install the required TileIR compiler components in the Python environment.
- Python version (project uses Python 3.12 in CI).
- Nsight Compute path/version.
- Fabric Manager presence/status on NVSwitch/HGX systems.
- Linux kernel, glibc, and host compiler information.
- Relevant CUDA environment variables.
- The profiling permission situation for the unprivileged GitHub Actions runner.

Run:

    python3 scripts/cutile_sanity_check.py

The script is read-only.
