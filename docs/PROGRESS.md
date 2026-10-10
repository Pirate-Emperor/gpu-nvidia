# Project Progress Log

This file is the durable experiment log for the project. Record what was tested, what passed, what failed, and what remains blocked.

## 2026-10-05 / 2026-10-06 — Foundation

### Repository foundation
- Added an independent FP32 NumPy mathematical reference.
- Added a tiled online-softmax NumPy reference.
- Added a deliberately unfused CuPy GPU baseline.
- Added a scalar CUDA baseline.
- Added an optional PyTorch SDPA reference.
- Added the first cuTile fused causal-attention kernel.
- Added required launch-contract validation.
- Added deterministic input generation and exact NPZ/NPY I/O.
- Added correctness/contract/I/O/generator/online-reference tests.
- Added a frozen representative benchmark matrix.
- Added a persistent human-readable test/validation ledger at docs/TEST_MATRIX.md.

### Expanded test coverage
- Added randomized finite-output coverage across B/H/N and D=32/64/128.
- Added exact prefix-mean checks for zero logits.
- Added constant-V invariance checks across all required head dimensions.
- Added extreme finite-logit stability checks.
- Added convex-hull invariants for attention outputs.
- Added explicit batch/head isolation checks.
- Added additional current-query locality checks.
- Added GPU-only tests for asymmetric legal tile configurations, tile-to-tile agreement, causal future-K/V invariance, constant-V behavior, zero-logit prefix means, nonfinite input rejection, non-contiguous rejection, and unsupported/mismatched dimensions.

### Independent native CUDA work
- Added a custom block-row fused CUDA kernel that uses ordinary CUDA C++ and cooperative reductions.
- Added a second warp-row fused CUDA kernel using warp-shuffle reductions.
- Added a third shared-memory tiled fused CUDA kernel with configurable 32/64 key tiles.
- Added independent correctness suites against the FP32 oracle and tiled online-softmax reference.
- Added cross-implementation tests between the two custom fused CUDA variants.
- Added a native-CUDA GitHub Actions workflow that deliberately does not require cuTile or R580+, so the R550 node can validate these implementations independently.
- Added a baseline benchmark path comparing unfused CuPy, scalar CUDA, block-row fused CUDA, warp-row fused CUDA, and shared-memory tiled CUDA.

### CPU validation
- The self-hosted A100 runner successfully executed the CPU/reference suite.
- The corrected suite reached 75 passing tests before this latest edge-case expansion.
- The newly added CPU tests are preserved and will run automatically through the repository test workflow.
- One initial test failure was a test-shape assertion, not an attention implementation failure; it was fixed and re-run successfully.

### A100 environment
- The self-hosted runner selects an NVIDIA A100-SXM4-40GB node.
- The node reports 40 GiB and compute capability 8.0.
- Current driver is R550.90.07.
- Current NVIDIA cuTile Python releases require a newer driver, so the production cuTile workflow stops at an explicit environment gate.
- A read-only access audit reports UID/EUID 55012, not root; sudo exists but passwordless sudo is unavailable and noninteractive sudo policy inspection requires credentials.
- Conclusion: a host-driver upgrade cannot be performed by the GitHub runner account itself. It requires an administrator/privileged maintenance path, unless the cluster provides a separately authorized upgrade mechanism.

### Forward-compatibility experiment
- NVIDIA cuda-compat-13-4 was downloaded from the NVIDIA repository without root.
- The compatibility libraries were extracted into the job workspace.
- Native CUDA 13.x access failed with cudaErrorInsufficientDriver.
- Loading the extracted compatibility libraries produced cudaErrorSystemDriverMismatch (CUDA error 803).
- Therefore the current R550 node has not been validated as a cuTile runtime through that compatibility configuration.

### cuTile 1.2 experiment
- A branch attempted to test an early cuTile release associated with Ampere/SM80 support.
- The currently reachable Python package index does not expose cuda-tile 1.2.0, so that branch did not reach GPU execution.

## Engineering infrastructure added
- Added a project-structure test that freezes required deliverables.
- Added a read-only driver-access audit workflow.
- Added benchmark and Nsight workflows that are ready for a supported A100/cuTile runtime.
- Added result-storage conventions and a persistent status snapshot.

## Baseline ladder
1. NumPy FP32 full attention.
2. NumPy tiled online-softmax.
3. Unfused CuPy attention.
4. Simple scalar CUDA attention.
5. Optional PyTorch SDPA reference.
6. Fused cuTile attention.

Each executable GPU backend has an independent correctness-test path against the mathematical oracle.

## Next execution gate
1. Ask the cluster administrator for an R580+ driver path or another supported A100 node.
2. Run the expanded GPU correctness matrix.
3. Capture at least two Nsight Compute profiles.
4. Run >=30-iteration benchmark sessions and preserve raw JSON/CSV.
5. Optimize only after correctness and profiling evidence exist.

## Experiment policy
For each material kernel change:
1. preserve/update the relevant tests;
2. record the exact environment and configuration;
3. record successful and unsuccessful outcomes;
4. re-run correctness before claiming a performance improvement;
5. preserve benchmark/profiler artifacts instead of replacing prior runs.


## 2026-10-06 — Current development direction

### Driver-independent engineering
The project is no longer blocked on cuTile for algorithmic validation. The custom
CUDA variants are intended to exercise the same fused mathematical path:
scaled QK scores -> causal restriction -> stable max-shifted softmax -> V
accumulation, with FP16 inputs and FP32 accumulation.

The native CUDA workflow is intentionally separate from the cuTile workflow.
A successful native-CUDA run would validate the ordinary CUDA implementations
on the existing A100/R550 node without implying that cuTile is supported.

### Local execution note
The development sandbox cannot resolve github.com, so the latest repository
state was not executed locally from this environment. CI remains the source of
truth for execution results. No unexecuted test or benchmark is marked PASS.

## 2026-10-06 — CI sanity finding and correction

The first CI execution after the expanded test additions completed 94 tests and
reported 5 failures plus 11 GPU/package skips. The failures were all in newly
added CPU/reference tests, not GPU execution:
- constant-value exact-equality assertions were too strict for FP32 reduction;
- one extreme-logit expected array had a missing leading B/H dimension;
- the common-key-shift metamorphic property used an overly strict tolerance
  despite FP16 input rounding.

These test specifications were corrected on main. The GPU audit run remains a
separate self-hosted A100 job and must not be inferred from this hosted CPU run.

## 2026-10-06 — Driver maintenance is deferred; development continues

The cluster administrator has confirmed that an R580+ driver upgrade is possible during the next scheduled maintenance cycle, but the cycle may be delayed and no other available node currently provides R580+.

Therefore the project intentionally continues without cuTile execution rather than waiting:
1. validate independent CUDA implementations on the existing A100/R550 host;
2. benchmark unfused and fused ordinary-CUDA variants;
3. validate both A100 devices exposed to the runner;
4. harden correctness, edge-case, I/O, and reproducibility tests;
5. collect native CUDA/Nsight evidence where the current software stack permits;
6. after the driver upgrade, install/validate cuTile and run the same frozen correctness/performance matrix.

Important: native CUDA results are development/control results. They are not presented as cuTile results and do not replace the required cuTile implementation.


## 2026-10-08 / 2026-10-09 — Regression freeze and optimization experiments

The consolidated evidence and performance tables live in [docs/EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md); use that report rather than reconstructing results from this chronological log.

### Confirmed correctness state
- A100 native CUDA run #44 on `a8172749`: **133 passed, 1 skipped**. It followed the fix to an invalid affine-property test that incorrectly requested unsupported `tile_k=2`; the kernel's supported tile sizes remain 16/32/64.
- Broad non-cuTile run #47 on the corrected test commit: **905 passed, 52 skipped**. Latest rerun #51: **905 passed, 52 skipped, 0 failed**.
- Latest CPU/reference run #285: **761 passed, 68 skipped, 0 failed**. Hosted CPU CI skips GPU tests when CuPy/GPU dependencies are not present.
- Dedicated v4 and v5 GPU correctness workflows: **30/30 passed each**. These variants remain experiments, not promoted defaults.

### Version ladder and compiler evidence
- Added `benchmarks/benchmark_tile_ladder.py`: v1/v2/v3/current comparisons across frozen cases and tile_k 16/32/64, with 10 warmups, 30 CUDA-event iterations, median/P90, and first-compile time measured separately.
- Ladder run #3 measured first compile as v1 1652 ms, v2 717 ms, v3 1063 ms, current 808 ms. Per-case kernel medians and limitations are documented in the experiment report.
- Added `scripts/ptxas_tile_report.py` and ran it on SM80. v1/v2/v3/current/v4 compiled without reported spills. v3 uses 40 registers; current and v4 use 32 registers and emitted identical reported shared-memory usage/cubin size.
- v4 tries persistent Q registers across K tiles. It passed correctness but had mixed timing, regressed a long-batched case, and generated code equivalent to current; it is not promoted.
- v5 tries `half2` QK loads. It passed 30 correctness tests; selected cases were faster, but per-run/per-tile timing variability and mixed regressions make the result inconclusive. Current remains the main A100 surrogate until a controlled repeat confirms a robust gain.

### Profiling and runtime blockers
- Native Nsight workflow detected Nsight Compute 2024.1.1 and both A100s, but profiling failed with `ERR_NVGPUCTRPERM`; no kernels/counters were profiled.
- CUDA 13.4 forward compatibility was not functional on R550 (insufficient-driver error / CUDA error 803). The CUDA 13.0 package/runtime attempt did not establish a verified working cuTile GPU run. Current cuTile execution remains blocked by the R580+ runtime gate.
- The ordinary CUDA A100 work is explicitly a cuTile-like surrogate, not actual NVIDIA cuTile.

### Documentation
- Added [docs/README.md](README.md) as the documentation map.
- Added [docs/EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the consolidated test ledger, all approach families, timing results, PTXAS evidence, infrastructure attempts, and next decisions.


## 2026-10-10 — Repeated v5 benchmark and full PTXAS comparison

### Compiler resource check
- Added `src/variants/shared_tile_v5.py` to `scripts/ptxas_tile_report.py`, enabled the workflow to archive the JSON output, and ran [PTXAS workflow #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) successfully.
- SM80 PTXAS reports zero spill loads/stores for v1/v2/v3/current/v4/v5. v5 uses 32 registers and 33,792 B shared memory, equal to current; v5's cubin is 15,520 B vs current's 15,136 B.

### Controlled v5 comparison
- Reworked `benchmarks/benchmark_v5.py` to compare current and v5 in paired sessions, alternating which version runs first. Each version/session uses 10 warmups and 30 CUDA-event samples; three repeats are recorded for every one of seven cases and each tile_k=16/32/64. Raw session JSON/CSV and a summary CSV are uploaded by [A100 v5 run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163).
- Correctness stage passed **30/30 tests** before timings. The run succeeded.
- Measured v5 gains were modest on medium D=64, mixed across short cases, and only about 0.3–0.6% on long/batched D=128. Ratios varied widely for some short cases. v5 remains unpromoted because the data do not establish a robust general win.
- Latest CPU/reference run [#306](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001131) passed **761 tests, 68 skipped**.

### Remaining blockers
- Actual cuTile still needs a supported driver/runtime and an actual successful compile/launch/correctness run.
- Nsight Compute counter profiling remains blocked by `ERR_NVGPUCTRPERM`; the successful PTXAS report is compile-time evidence only, not hardware-counter profiling.


## 2026-10-10 — Nsight Systems timelines for the A100 CUDA surrogate

- Added `.github/workflows/a100-nsys-timeline.yml` to capture two warmed ordinary-CUDA shared-tile launches without requiring Nsight Compute hardware counters.
- [Workflow #1](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055492320) completed successfully on A100/R550. The runner/version check and both capture commands passed; artifact upload succeeded.
- Uploaded [artifact #11671195371](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055492320/artifacts/11671195371) containing two Nsight Systems reports: medium D=128 with tile_k=64, and longer batched D=128 with tile_k=32.
- This provides CUDA API/kernel timeline evidence for the ordinary-CUDA surrogate, not cuTile and not Nsight Compute hardware-counter metrics. Actual-cuTile profiling remains open.
