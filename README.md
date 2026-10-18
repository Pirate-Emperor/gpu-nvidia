# Fused Causal Attention with NVIDIA cuTile

IIT Madras GPU Programming course project: implement and optimize causal scaled dot-product self-attention as a fused NVIDIA cuTile GPU kernel, targeted at NVIDIA A100/SM80.


## Start here: project navigation

Use this order to understand the project and decide what to do next.

| Purpose | Document |
|---|---|
| **What remains to do / completion gates** | [TODO checklist](docs/TODO_CHECKLIST.md) |
| Documentation map and recommended reading order | [Documentation index](docs/README.md) |
| Approaches, benchmark evidence, failure analysis, and decisions | [Experiment report](docs/EXPERIMENT_REPORT.md) |
| Correctness contract, tolerances, and regression expectations | [Testing contract](docs/TESTING_CONTRACT.md) |
| Coverage matrix and latest test status | [Test matrix](docs/TEST_MATRIX.md) |
| Ordinary-CUDA implementation differences and variant decisions | [CUDA variants](docs/CUDA_VARIANTS.md) |
| Reference/baseline comparison plan | [Baseline ladder](docs/BASELINE_LADDER.md) |
| Driver, toolkit, compatibility, and profiler constraints | [Environment notes](docs/ENVIRONMENT.md) |
| External cluster prerequisites and admin-side checklist | [Admin checklist](docs/ADMIN_CHECKLIST.md) |
| Dated experiment history | [Progress log](docs/PROGRESS.md) |
| How benchmark/profile artifacts should be stored | [Results README](results/README.md) |
| Current hardware, test, and blocker snapshot | [Results status](results/STATUS.md) |
| Full handoff for a new chat: runs, benchmarks, environment, security plan, and next steps | [Chat handoff](docs/CHAT_HANDOFF_2026-10-18.md) |
| Copy-paste prompt to continue in a new chat | [Continuation prompt](docs/CONTINUE_PROMPT.md) |

**Main code and experiment entry points:** [actual cuTile target](src/cutile_attention.py) · [current shared-tile CUDA surrogate](src/shared_tile_cuda_attention.py) · [unfused CuPy baseline](src/baseline.py) · [primary correctness tests](tests/) · [benchmark directory](benchmarks/) · [GitHub Actions workflows](.github/workflows/).

> **Naming reminder:** the shared-tile CUDA versions are ordinary CUDA C++ implementations and performance/correctness controls. They are not actual NVIDIA cuTile runs.

## What cuTile is — and where this project stands

[NVIDIA cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) is a tile-based GPU programming model: we describe operations on blocks of array data, while the cuTile compiler/runtime maps those tile operations to GPU execution. Our assignment is to use this model to write a fused causal-attention forward kernel.

**What is already done:** independent mathematical references, broad ordinary-CUDA correctness coverage, several CUDA comparison kernels, preserved v1–v5 experiments, a paired benchmark ladder, PTXAS summaries, Nsight Systems traces for the surrogate, and a consolidated experiment report. Latest verified: broad A100 non-cuTile run #52, 907 passed / 52 skipped; CPU/reference run #340, 763 passed / 68 skipped. These are not actual cuTile results.

**What is not done yet:** actual `src/cutile_attention.py` execution and full correctness on a supported GPU/runtime, controlled actual-cuTile performance comparisons, and successful Nsight Compute profiles. The available A100 runner uses driver R550.90.07, and the project's current cuTile environment gate requires R580+; its profiler also reports `ERR_NVGPUCTRPERM`. Do not count the ordinary-CUDA surrogate results as completed cuTile results.

**Execution decision (2026-10-18):** stop using the existing A100 runner as the actual-cuTile route. Its verified R550.90.07 driver is below NVIDIA's documented R580 minimum, and prior CUDA compatibility attempts did not produce a verified cuTile run. Keep that A100 runner for ordinary-CUDA surrogate tests only. Move actual-cuTile bring-up to the user's local GPU after its real hardware/OS/toolchain inventory, or to another explicitly authorized GPU environment meeting NVIDIA's current requirements. Do not assume an RTX 4060 or any local driver/toolkit state.

**Immediate next action:** collect the local machine inventory in "Next objectives" below. Choose Windows-native versus Linux/WSL2 only after seeing the actual OS/GPU/driver output. Do not register a runner or claim isolation until the OS-enforced security boundary is matched to and tested on that environment. See the [TODO checklist](docs/TODO_CHECKLIST.md) and [environment notes](docs/ENVIRONMENT.md).

## Goal

The required kernel must:

- consume FP16 Q/K/V;
- accumulate in FP32;
- implement causal self-attention only;
- fuse score computation, scaling, causal masking, stable online softmax, and V accumulation;
- avoid materializing the complete score/probability matrix in global memory;
- support the required head dimensions D = 32, 64, 128;
- require tile-divisible sequence lengths for the selected tile sizes.

The course proposal additionally requires at least two tile configurations, deterministic correctness tests, an unfused baseline, a framework/vendor performance reference, >=30 measured iterations, separate JIT reporting, and profiler evidence.

## Repository layout

src/reference.py
  Independent FP32 NumPy mathematical oracle.

src/online_reference.py
  Independent tiled online-softmax recurrence.

src/baseline.py
  Deliberately unfused CuPy GPU baseline that materializes score/probability tensors.

src/naive_cuda_attention.py
  Simple scalar CUDA baseline. It is intentionally inefficient and exists to establish a basic CUDA performance floor.

src/custom_cuda_attention.py
  Independent fused CUDA implementation using ordinary CUDA C++ and cooperative block reductions; no cuTile dependency.

src/warp_cuda_attention.py
  Independent fused CUDA implementation using warp-shuffle reductions; no cuTile dependency.

src/shared_tile_cuda_attention.py
  Current A100 shared-memory tiled fused CUDA surrogate; no cuTile dependency.

src/variants/shared_tile_v1.py ... shared_tile_v5.py
  Preserved optimization ladder: one-row two-pass → warp-row online softmax →
  four-score loop → Q-register experiment → half2 QK experiment.

docs/CUDA_VARIANTS.md
  Design notes and invariants for the independent CUDA fused implementations.

docs/ADMIN_CHECKLIST.md
  Cluster-side hardware/software/profiling prerequisites before cuTile execution.

src/torch_reference.py
  Optional PyTorch SDPA reference/performance backend.

src/cutile_attention.py
  Main cuTile fused implementation.

tests/
  Correctness and contract tests.

benchmarks/benchmark_attention.py
  Fused-vs-unfused benchmark using GPU events.

benchmarks/profile_fused.py
  Warmed single-launch target for Nsight Compute.

docs/BASELINE_LADDER.md
  Comparison strategy.

docs/TESTING_CONTRACT.md
  Frozen correctness contract.

docs/PROGRESS.md
  Persistent chronological experiment log.

docs/README.md
  Documentation index and recommended reading order.

docs/EXPERIMENT_REPORT.md
  Consolidated approaches, exact CI outcomes, measured latency, compiler
  resources, infrastructure blockers, and next decisions.

results/
  Raw benchmark and profiler artifacts.

## CPU validation

CPU/reference tests need only NumPy and pytest:

    python -m pip install -r requirements-dev.txt
    pytest -q -m "not gpu"

The CPU suite does not validate CUDA execution.

## Native CUDA validation

A separate self-hosted A100 workflow runs the ordinary CUDA baselines without
importing or requiring cuTile. This is intentional: the current R550 node can
still be used to validate the fused CUDA algorithm while the cuTile environment
is blocked.

The native CUDA ladder includes the scalar CUDA baseline, a block-row fused
kernel, a warp-row fused kernel, and the unfused CuPy baseline. Every fused
variant is checked against the independent NumPy oracle and the tiled
online-softmax reference.

## GPU environment

Internet-dependent installation on the project runner must go through the root-level core_config:

    python3 scripts/with_core_config.py <command> ...

This imports the user's root-level core_config and preserves its configured proxy environment without storing proxy credentials in the repository.

## Required CLI

    python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile

## Correctness matrix

The GPU target is checked against the independent FP32 NumPy oracle with a documented mixed tolerance.

Coverage includes:

- D = 32, 64, 128.
- Short, medium, and long sequences.
- Multiple head counts and batch sizes.
- Multiple tile configurations.
- Small and large score magnitudes.
- Causal future-token invariance.
- Rejection of non-divisible sequence lengths.
- NaN/Inf checks.

## Baseline ladder

1. NumPy FP32 oracle.
2. NumPy tiled online-softmax reference.
3. Unfused CuPy GPU baseline.
4. Simple scalar CUDA baseline.
5. Custom block-row fused CUDA.
6. Custom warp-row fused CUDA.
7. Custom shared-memory tiled CUDA.
8. Optional PyTorch SDPA reference.
9. Fused cuTile target.

This lets us distinguish mathematical correctness from GPU-kernel optimization.

## Performance method

Steady-state measurements use GPU events, warm-up iterations, >=30 measured iterations, median/P90 latency, speedup versus the deliberately unfused baseline, effective output throughput, explicit GPU/toolchain metadata, and separate cold/JIT timing.

Profiler runs use a dedicated single-kernel target so Nsight Compute measurements are not polluted by the entire benchmark harness.

## Current status (last checked 2026-10-18; most recent successful runs are dated 2026-10-10)

- Latest broad non-cuTile regression: **907 passed, 52 skipped, 0 failed** ([run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082)).
- Latest CPU/reference suite: **763 passed, 68 skipped, 0 failed** ([run #340 on pre-handoff main SHA afe530b](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056390931)).
- Dedicated shared-tile v4 and v5 A100 correctness: **30/30 passed each**; neither experiment is promoted as a guaranteed performance win.
- Repeated current-vs-v5 benchmark: [run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163), three paired/order-alternating trials for every case/tile; v5 remains unpromoted due to mixed results and sub-1% long/batched gains.
- Updated PTXAS comparison: [run #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) compiled v1–v5 plus current with zero reported spills; v5 cubin is 384 B larger than current.
- Nsight Systems timeline: [run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950) captured two warmed ordinary-CUDA surrogate traces and CUDA API/kernel summaries ([artifact](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954)); actual-cuTile profiling remains open.
- Actual cuTile execution remains blocked by driver R550.90.07; current cuTile environment gate requires R580+.
- Nsight Compute is installed, but the runner lacks performance-counter permission (`ERR_NVGPUCTRPERM`), so no real counter profile was captured.

**Actions evidence access note (2026-10-18 continuation):** repository contents are readable, but the connected GitHub Actions run list currently returns zero visible runs; direct run-detail, job and artifact requests for the handoff run IDs return 404, and the current handoff commit has no visible combined statuses. This does **not** prove the historical runs do not exist. Their counts below remain values recorded in the previous handoff and could not be independently re-audited from this connection.

**Do not confuse the native-CUDA/cuTile-like surrogate with actual NVIDIA cuTile.** The detailed evidence, approach comparison, test results, benchmarks and limitations are in [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md). See [docs/README.md](docs/README.md) for the documentation map.


## Next objectives (do these in order)

1. **Inventory the local GPU before configuring a runner.** On Windows PowerShell, run:

       nvidia-smi
       nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,memory.free,pci.bus_id --format=csv
       nvcc --version

   If a query field is unsupported, use `nvidia-smi -q`. Also report OS/build or Linux distro/kernel, WSL2 status, Python version and Docker availability. On Windows PowerShell: `Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber, OsArchitecture`, `wsl --status`, `python --version`, `docker --version`. On Linux/WSL2: `cat /etc/os-release`, `uname -a`, `python3 --version`, and whether `docker` and `nvcc` are present. Missing `nvcc` is useful information; the CUDA version shown by `nvidia-smi` is not proof that NVCC/the toolkit is installed. Keep credentials and machine secrets out of the output.
2. **Create a least-privilege, project-only GPU runner.** Prefer a disposable Linux VM or a tightly isolated GPU-enabled container. Use a dedicated unprivileged account, one writable project workspace, no sudo/administrator role, no personal home-directory mounts, no host Docker socket, no privileged container, no personal tokens/SSH keys/cloud credentials, minimal outbound network, and an ephemeral runner when feasible. Restrict it to this private repository and trusted/manual workflows; do not run untrusted pull-request code on it. The runner itself must authenticate to GitHub, but its registration/connection credentials must stay local and never be pasted into chat or exposed to job steps. ChatGPT does not gain interactive shell access merely because a runner is registered; Actions workflows run commands on the runner.
3. **Prove the environment first.** Check driver, GPU visibility, toolkit/compiler and Python environment, then compile and launch a tiny actual cuTile smoke kernel and compare its output. Do not count imports or skipped tests as successful GPU execution.
4. **Run actual-cuTile correctness, then benchmark.** Exercise src/cutile_attention.py, D=32/64/128, causal/stability/edge invariants, two or more legal tile configurations, and the CLI. Only after correctness passes, capture cold/JIT time separately and run CUDA-event timing with at least 10 warmups and at least 30 timed iterations.
5. **Complete hardware profiling.** Obtain explicitly authorized Nsight Compute counter access, then save at least two real actual-cuTile profiles (medium and long/batched). The existing Nsight Systems traces cover only ordinary CUDA.
6. **Optimize based on evidence and update all result docs.** Keep current/v5 performance claims conservative, preserve regressions and artifacts, and update the experiment report, test matrix, progress log, and status snapshot after each meaningful run.

The [handoff document](docs/CHAT_HANDOFF_2026-10-18.md) contains exact prior results, benchmark numbers, runner details, the secure setup checklist, and a ready-to-paste continuation prompt.

## Course-scope boundary

The following are exploratory only and should be postponed until the required path is correct, benchmarked, and profiled:

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache, packed variable-length sequences, dropout, partial edge tiles, and alternative DSL implementations.
