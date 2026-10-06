# GPU Environment Contract

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
