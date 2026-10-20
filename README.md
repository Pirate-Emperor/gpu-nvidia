# Fused Causal Attention with NVIDIA cuTile

IIT Madras GPU Programming course project: implement and optimize causal scaled dot-product self-attention as a fused NVIDIA cuTile GPU kernel. The current actual-cuTile bring-up target is the inventoried GeForce RTX 4060 (SM89) on native Windows; the existing A100/R550 machine remains an ordinary-CUDA control only.


## Start here: project navigation

Use this order to understand the project and decide what to do next.

| Purpose | Document |
|---|---|
| **What remains to do / completion gates** | [TODO checklist](docs/TODO_CHECKLIST.md) |
| **Two-page project summary and current gates** | [Project two-pager](docs/PROJECT_TWO_PAGER.md) |
| Rubric-mapped course checklist | [Course project checklist](docs/COURSE_PROJECT_COMPLETION_CHECKLIST.md) |
| **Complete experiment narrative** | [Root Report.md](Report.md) |
| Automation-only workflow provenance | [Run evidence (separate file)](RUN_EVIDENCE.md) |
| Current course project technical report (Markdown source and PDF) | [20 October report](docs/COURSE_PROJECT_REPORT_2026-10-20.md) · [downloadable PDF](docs/COURSE_PROJECT_REPORT_2026-10-20.pdf) |
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
| **Latest dated RTX 4060 benchmark/accuracy/hint snapshot** | [20 October results](docs/CUTILE_RTX4060_RESULTS_2026-10-20.md) |
| Actual-cuTile Nsight Compute profiling results | [Profiler analysis](docs/CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) |
| Local RTX 4060/Windows cuTile bring-up and smoke evidence | [Local GPU guide](docs/LOCAL_GPU_WINDOWS.md) |
| Native Windows RTX 4060 Actions runner setup | [Local GPU guide](docs/LOCAL_GPU_WINDOWS.md) |
| Full handoff for a new chat: runs, benchmarks, environment, security plan, and next steps | [Chat handoff](docs/CHAT_HANDOFF_2026-10-18.md) |
| Copy-paste prompt to continue in a new chat | [Continuation prompt](docs/CONTINUE_PROMPT.md) |

**Main code and experiment entry points:** [actual cuTile target](src/cutile_attention.py) · [current shared-tile CUDA surrogate](src/shared_tile_cuda_attention.py) · [unfused CuPy baseline](src/baseline.py) · [primary correctness tests](tests/) · [benchmark directory](benchmarks/) · [GitHub Actions workflows](.github/workflows/).

> **Naming reminder:** the shared-tile CUDA versions are ordinary CUDA C++ implementations and performance/correctness controls. They are not actual NVIDIA cuTile runs.

## What cuTile is — and where this project stands

NVIDIA cuTile Python is a tile-based GPU programming model. This project implements fused causal scaled dot-product self-attention as one cuTile kernel, keeping the complete score/probability matrix out of global memory.

**Latest complete actual-cuTile validation:** [native Windows RTX 4060 workflow #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), commit `bf5a6233e195c2ced647b4e20b5b2d914da60e85`, passed the genuine cuTile smoke test, **48 core attention tests**, **3 GQA/MQA and packed-varlen GPU tests**, **29 numerical case/tile configurations with zero tolerance violations**, and CLI output verification against the independent NumPy oracle. Three controlled benchmark sessions used ten warmups, 30 CUDA-event samples, five launches per sample, and rotated tile order. Raw results and logs are preserved in [artifact #12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195).

**Current optimization:** the fused kernel skips wholly-future K/V tiles in the causal loop and uses FP16 probability operands for P×V with FP32 accumulation and FP32 online-softmax state. The 64×64 tile is the default; 64×128 was the fastest configuration observed for the longer batched D=128 case, pending another independent repeat before calling it a general winner.

**Current core status:** the latest complete RTX 4060 run above validates the core plus selected GQA/MQA and packed-sequence gates on the current cuTile kernel. The repeat run [#39423190965](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190965) passed setup, genuine cuTile smoke, and the required core suite, but stopped during the optional GQA/MQA/packed-varlen step and produced no retrievable log/artifact, so it is not counted as a complete pass. A targeted NCU hint-10 capture and an isolated CUDA 13.2 PyTorch/cuBLAS diagnostic are queued. The existing PyTorch CUDA 13.0 wheel fails a plain `torch.bmm` smoke before the cuTile bridge, so no PyTorch SDPA timing or speedup claim is made. The checklist still tracks a matched NCU review, clean-cache JIT coverage for long specializations, framework comparison, and confirmed individual contributions. Existing measured speedups are versus the deliberately unfused FP32 CuPy baseline.

See the [detailed RTX 4060 results and full tile matrix](docs/CUTILE_RTX4060_RESULTS_2026-10-19.md), [experiment report](docs/EXPERIMENT_REPORT.md), [test matrix](docs/TEST_MATRIX.md), and [TODO checklist](docs/TODO_CHECKLIST.md). Use [Report.md](Report.md) as the full experiment narrative, the [rubric-mapped checklist](docs/COURSE_PROJECT_COMPLETION_CHECKLIST.md) for milestone tracking, and [RUN_EVIDENCE.md](RUN_EVIDENCE.md) for workflow-only provenance.

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

## Current status (verified 2026-10-19)

- **Actual cuTile correctness:** PASS on RTX 4060. Workflow #39160827910, commit 12120f8a67a1fbd6f9cd83bd8f2f8bf7b31ce428: 48 passed, 0 failed with REQUIRE_CUTILE=1.
- **Numerical diagnostics:** PASS for 29 case/tile configurations; maximum absolute error 0.0009765625, zero violations of the frozen mixed tolerance.
- **CLI / independent oracle:** PASS; output is FP16 and passed shape, dtype, finite-value, and reference checks.
- **Controlled performance:** three alternating-order sessions over five tile configurations. The long D=128 case measured 0.267059 ms / 16.53× vs unfused CuPy on 64×64; longer batched D=128 measured 0.923034 ms / 18.54× on 64×128. Short cases show large Windows/WDDM timing noise and are not headline speedup claims.
- **Nsight Compute profiling:** PASS. Two actual-cuTile kernel configurations (medium D=128 at 64×64 and longer-batched D=128 at 64×128) were captured and inspected. [Profile artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039); [analysis](docs/CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md).

The raw benchmark logs and JSON/CSV are in [artifact 12148414465](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910/artifacts/12148414465). The two actual-cuTile profiler configurations are preserved in [profile artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039), with the interpretation in [the profile analysis](docs/CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md). The core correctness, performance, and profiling gates are complete; the CUDA-enabled PyTorch SDPA comparison remains an optional follow-up.

## Follow-up opportunities (not blockers for the validated core scope)

1. Consider a focused shared-memory/register-pressure experiment: the profiles show one CTA per SM and roughly 8.3% occupancy for the two sampled configurations. Keep the current kernel unless a new version passes correctness and controlled benchmarking.
2. Optionally add a CUDA-enabled PyTorch SDPA comparison if a compatible build can be installed safely in the project venv. Do not confuse the existing speedup versus unfused CuPy with a speedup versus vendor-optimized attention.
3. Keep the raw profiler artifact and [profile analysis](docs/CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) alongside the benchmark evidence. Nsight’s more detailed stall-sampling metric was unavailable, so do not overinterpret its estimated local-speedup heuristics.

## Course-scope boundary

The following are exploratory only and should be postponed until the required path is correct, benchmarked, and profiled:

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache, packed variable-length sequences, dropout, partial edge tiles, and alternative DSL implementations.
