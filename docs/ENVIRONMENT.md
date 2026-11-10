# Environment notes

## Actual cuTile target

- GPU: GeForce RTX 4060, compute capability 8.9.
- OS: Windows 11 x64.
- Driver: 616.92.
- Python: 3.12.10.
- NumPy: 2.5.4.
- CuPy: 14.2.0.
- cuda-tile: 1.6.0.
- TileIRAS, NVCC and NVVM Python packages: 13.4.92.

These versions were recorded during the October 2026 validation. Check the installed environment before repeating a run.

## Driver and compiler

cuTile requires a supported GPU and a sufficiently recent NVIDIA driver. The checked setup uses driver 616.92 and a project-local Python environment. The pip TileIRAS/NVCC/NVVM packages supply the compiler path; a system nvcc is not required for this environment.

The three compiler packages must use matching major/minor versions. See the [NVIDIA cuTile Python quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) for current requirements.

## A100 boundary

The Linux A100 machine has driver R550.90.07. It is used for ordinary CUDA C++ comparisons. Do not describe those runs as cuTile: that driver is below the cuTile driver requirement.

## Before a GPU run

- Check nvidia-smi and confirm the expected GPU and driver.
- Use the project virtual environment.
- Run the cuTile runtime check, then the 48-test suite with REQUIRE_CUTILE=1.
- Keep logs, benchmark results and profiler reports with the commit and machine details.
- Do not change the global CUDA installation to solve a project-venv issue.

The full setup history and failed environment experiments remain in Report.md and RUN_EVIDENCE.md.
