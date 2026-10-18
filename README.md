# Fused Causal Attention with NVIDIA cuTile

IIT Madras GPU Programming course project: implement and optimize causal scaled dot-product self-attention as a fused NVIDIA cuTile GPU kernel. The current actual-cuTile bring-up target is the inventoried GeForce RTX 4060 (SM89) on native Windows; the existing A100/R550 machine remains an ordinary-CUDA control only.


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
| Local RTX 4060/Windows cuTile bring-up and smoke evidence | [Local GPU guide](docs/LOCAL_GPU_WINDOWS.md) |
| Dockerized RTX 4060 Actions runner setup | [Docker runner guide](docs/CUTILE_DOCKER_RUNNER.md) |
| Full handoff for a new chat: runs, benchmarks, environment, security plan, and next steps | [Chat handoff](docs/CHAT_HANDOFF_2026-10-18.md) |
| Copy-paste prompt to continue in a new chat | [Continuation prompt](docs/CONTINUE_PROMPT.md) |

**Main code and experiment entry points:** [actual cuTile target](src/cutile_attention.py) · [current shared-tile CUDA surrogate](src/shared_tile_cuda_attention.py) · [unfused CuPy baseline](src/baseline.py) · [primary correctness tests](tests/) · [benchmark directory](benchmarks/) · [GitHub Actions workflows](.github/workflows/).

> **Naming reminder:** the shared-tile CUDA versions are ordinary CUDA C++ implementations and performance/correctness controls. They are not actual NVIDIA cuTile runs.

## What cuTile is — and where this project stands

[NVIDIA cuTile Python](https://docs.nvidia.com/cuda/cutile-python/) is a tile-based GPU programming model: we describe operations on blocks of array data, while the cuTile compiler/runtime maps those tile operations to GPU execution. Our assignment is to use this model to write a fused causal-attention forward kernel.

**What is already done:** independent mathematical references, broad ordinary-CUDA correctness coverage, several CUDA comparison kernels, preserved v1–v5 experiments, a paired benchmark ladder, PTXAS summaries, Nsight Systems traces for the surrogate, and a consolidated experiment report. Latest verified: broad A100 non-cuTile run #52, 907 passed / 52 skipped; CPU/reference run #340, 763 passed / 68 skipped. These are not actual cuTile results.

**What is not done yet:** actual `src/cutile_attention.py` execution and full correctness on a supported GPU/runtime, controlled actual-cuTile performance comparisons, and successful Nsight Compute profiles. The available A100 runner uses driver R550.90.07, and the project's current cuTile environment gate requires R580+; its profiler also reports `ERR_NVGPUCTRPERM`. Do not count the ordinary-CUDA surrogate results as completed cuTile results.

**Execution decision (2026-10-18):** stop using the existing A100 runner as the actual-cuTile route; its R550.90.07 driver is below NVIDIA's documented R580 minimum. The user's local RTX 4060 is now inventoried at compute capability 8.9 and driver 616.92 on 64-bit Windows. That GPU/driver/OS combination meets the documented cuTile prerequisites on paper. Native Windows cuTile smoke now passes on the RTX 4060: cuTile 1.6.0 with matching CUDA 13.4 TileIRAS/NVCC/NVVM components compiled and launched a genuine vector-add kernel, synchronized, and exactly validated 4,096 outputs. This is a runtime smoke pass only; actual causal attention remains unvalidated. Docker Desktop with its WSL2 backend is now running and Docker Engine/Compose are confirmed working. The first runner launch started the container but did not confirm that the runner reached Online. The next step is to start the persistent Dockerized runner using an owner-supplied repository registration token, then manually dispatch the main-only validation workflow. See [docs/CUTILE_DOCKER_RUNNER.md](docs/CUTILE_DOCKER_RUNNER.md).

**Immediate next action:** run `git pull --ff-only origin main`, generate a repository runner registration token under GitHub Settings > Actions > Runners, and pass it locally to `scripts/start_cutile_docker_runner.ps1 -RunnerToken $plainToken`. The launcher checks that a local Linux container sees the RTX 4060, starts a persistent runner, and leaves workflow dispatch to you. Keep the token out of chat and shell history. Attention is not complete until correctness passes, controlled benchmarks are recorded, and two actual cuTile profiles are captured.

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
- Actual cuTile on the A100 runner remains blocked by R550.90.07. The local RTX 4060 (CC 8.9, driver 616.92, Windows x64) has passed the genuine vector-add cuTile smoke test; actual causal-attention correctness, benchmarks, and profiler captures remain pending. Docker Desktop/WSL2 are still required before the new ephemeral GPU Actions workflow can run.
- Nsight Compute is installed, but the runner lacks performance-counter permission (`ERR_NVGPUCTRPERM`), so no real counter profile was captured.

**Actions evidence access note (2026-10-18 continuation):** repository contents are readable, but the connected GitHub Actions run list currently returns zero visible runs; direct run-detail, job and artifact requests for the handoff run IDs return 404, and the current handoff commit has no visible combined statuses. This does **not** prove the historical runs do not exist. Their counts below remain values recorded in the previous handoff and could not be independently re-audited from this connection.

**Do not confuse the native-CUDA/cuTile-like surrogate with actual NVIDIA cuTile.** The detailed evidence, approach comparison, test results, benchmarks and limitations are in [docs/EXPERIMENT_REPORT.md](docs/EXPERIMENT_REPORT.md). See [docs/README.md](docs/README.md) for the documentation map.


## Next objectives (do these in order)

1. **Set up the Docker GPU runner.** The local genuine cuTile smoke test already passed. Install/enable WSL2, install Docker Desktop and GitHub CLI, authenticate with `gh`, then run `scripts/start_cutile_docker_runner.ps1`. It checks GPU passthrough and automatically dispatches/watches the main-branch correctness workflow.
2. **Pass the actual cuTile attention correctness matrix.** The Docker workflow sets `REQUIRE_CUTILE=1`, so missing dependencies cannot silently skip tests. Fix collection/runtime failures in the repository and rerun the workflow; record its artifact/log and exact commit. Do not set `REQUIRE_A100` on the RTX 4060.
3. **Validate CLI I/O end-to-end.** Run `python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile`; verify output shape, FP16 dtype, finite values and independent-reference tolerance.
4. **Benchmark only after correctness passes.** Cover D=32/64/128, at least two legal tiles and frozen cases; compare with unfused CuPy and a suitable framework baseline where available. Use at least 10 warmups and at least 30 CUDA-event samples, separately report cold/JIT, per-case median/P90, throughput, speedup, and raw JSON/CSV.
5. **Profile on supported tooling.** Confirm Nsight Compute is installed and its counters are available without privileged workarounds. The current driver exceeds the documented Windows detailed-profiling minimum, but the profiler installation/access itself is not yet verified. Preserve at least two actual-cuTile profile files and logs.
6. **Run the local Dockerized Actions workflow.** Follow [docs/CUTILE_DOCKER_RUNNER.md](docs/CUTILE_DOCKER_RUNNER.md). It uses an ephemeral non-root container without host bind mounts or a Docker socket, and only runs the manual workflow from `main`. Container isolation reduces exposure but is not a perfect security boundary. Keep registration credentials local.

The [handoff document](docs/CHAT_HANDOFF_2026-10-18.md) contains historical benchmark data and the remaining course completion gates.
 
## Course-scope boundary

The following are exploratory only and should be postponed until the required path is correct, benchmarked, and profiled:

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache, packed variable-length sequences, dropout, partial edge tiles, and alternative DSL implementations.
