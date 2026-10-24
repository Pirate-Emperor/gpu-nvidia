# GPU environment checklist

Updated: 24 October 2026.

## Which GPU to use

The real cuTile target is the Windows RTX 4060, compute capability 8.9, driver 616.92. The Linux A100 has driver R550.90.07, below the cuTile R580+ requirement, so its runs are ordinary CUDA C++ controls only.

## Requirements for an A100 cuTile host

- NVIDIA driver R580 or later; use R580.126.09 or later for detailed Nsight Compute profiling.
- Matching NVIDIA Fabric Manager on HGX/NVSwitch systems.
- Python 3.10–3.14; the project uses Python 3.12.
- A supported cuTile compiler setup. The project venv can supply TileIRAS, NVCC and NVVM packages; their major/minor versions must match.
- Nsight Compute CLI (ncu) for profiling.
- Permission to read GPU performance counters. Without it, NCU may report ERR_NVGPUCTRPERM.

See the [NVIDIA cuTile quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) for current requirements.

## Before a benchmark

- Use one exclusive GPU where possible.
- Record the GPU UUID, topology, clocks, power limit, temperature and utilization.
- Prefer a full GPU over a MIG slice for comparable measurements.
- Do not force clocks unless cluster policy allows it.
- Check that another profiler or DCGM job is not holding the performance counters.

## Read-only checks

~~~bash
python3 scripts/cutile_sanity_check.py
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
nvcc --version
ncu --version
systemctl is-active nvidia-fabricmanager
~~~

The final command applies only on hosts that use Fabric Manager. nvidia-smi alone does not prove that the cuTile compiler can build and run a kernel.
