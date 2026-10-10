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

**Main code and experiment entry points:** [actual cuTile target](src/cutile_attention.py) · [current shared-tile CUDA surrogate](src/shared_tile_cuda_attention.py) · [unfused CuPy baseline](src/baseline.py) · [primary correctness tests](tests/) · [benchmark directory](benchmarks/) · [GitHub Actions workflows](.github/workflows/).

> **Naming reminder:** the shared-tile CUDA versions are ordinary CUDA C++ implementations and performance/correctness controls. They are not actual NVIDIA cuTile runs.

## What cuTile is — and where this project stands

[NVIDIA cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) is a tile-based GPU programming model: we describe operations on blocks of array data, while the cuTile compiler/runtime maps those tile operations to GPU execution. Our assignment is to use this model to write a fused causal-attention forward kernel.

**What is already done:** independent mathematical references, broad correctness coverage, several ordinary-CUDA comparison kernels, preserved v1–v5 experiments, a benchmark ladder, PTXAS summaries, and a consolidated experiment report. The latest recorded broad non-cuTile regression is 905 passed / 52 skipped; the CPU/reference suite is 761 passed / 68 skipped.

**What is not done yet:** actual `src/cutile_attention.py` execution and full correctness on a supported GPU/runtime, controlled actual-cuTile performance comparisons, and successful Nsight Compute profiles. The available A100 runner uses driver R550.90.07, and the project's current cuTile environment gate requires R580+; its profiler also reports `ERR_NVGPUCTRPERM`. Do not count the ordinary-CUDA surrogate results as completed cuTile results.

**Next action:** open [the TODO checklist](docs/TODO_CHECKLIST.md) and work through Priority 0 first. The checklist explains each completed work area and defines the evidence needed to close every remaining gate.

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

## Current status (2026-10-10)

- Latest broad non-cuTile regression: **907 passed, 52 skipped, 0 failed** ([run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082)).
- Latest CPU/reference suite: **763 passed, 68 skipped, 0 failed** ([run #334](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264023)).
- Dedicated shared-tile v4 and v5 A100 correctness: **30/30 passed each**; neither experiment is promoted as a guaranteed performance win.
- Repeated current-vs-v5 benchmark: [run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163), three paired/order-alternating trials for every case/tile; v5 remains unpromoted due to mixed results and sub-1% long/batched gains.
- Updated PTXAS comparison: [run #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) compiled v1–v5 plus current with zero reported spills; v5 cubin is 384 B larger than current.
- Nsight Systems timeline: [run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950) captured two warmed ordinary-CUDA surrogate traces and CUDA API/kernel summaries ([artifact](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954)); actual-cuTile profiling remains open.
- Actual cuTile execution remains blocked by driver R550.90.07; current cuTile environment gate requires R580+.
- Nsight Compute is installed, but the runner lacks performance-counter permission (`ERR_NVGPUCTRPERM`), so no real counter profile was captured.

**Do not confuse the native-CUDA/cuTile-like surrogate with actual NVIDIA cuTile.** The detailed evidence, approach comparison, test results, benchmarks and limitations are in [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md). See [docs/README.md](docs/README.md) for the documentation map.

## Course-scope boundary

The following are exploratory only and should be postponed until the required path is correct, benchmarked, and profiled:

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache, packed variable-length sequences, dropout, partial edge tiles, and alternative DSL implementations.
