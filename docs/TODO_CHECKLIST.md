# Project TODO Checklist — Fused Attention with cuTile

**Last reviewed:** 2026-10-09  
**Purpose:** One actionable checklist for the course deliverable. Keep this file and the root [README](../README.md) current as work changes. Check an item only after there is evidence; a surrogate CUDA result must never be counted as actual cuTile validation.

**Status key:** `[x]` complete with evidence · `[ ]` open · **BLOCKED** requires external hardware, permission, or environment changes.

## At a glance

- **Engineering foundation:** complete — independent references, native-CUDA comparison kernels, extensive correctness tests, versioned experiments, and benchmark harnesses exist.
- **Native CUDA surrogate:** validated on A100/R550; broad regression is green.
- **Actual NVIDIA cuTile execution:** **BLOCKED** on the currently available A100 runner (driver R550.90.07; project runtime gate requires R580+).
- **Nsight Compute hardware-counter evidence:** **BLOCKED** by `ERR_NVGPUCTRPERM`.
- **Course deliverable:** not complete until the actual cuTile path is run, benchmarked, and profiled on a supported setup.

## 1. Completed — do not redo without new evidence

- [x] Freeze the required scope: causal self-attention forward; FP16 Q/K/V; FP32 accumulation; D=32/64/128; stable online softmax; no global N×N score/probability tensors in the fused implementation.
- [x] Provide independent correctness references: scalar FP64 oracle, full-matrix FP32 NumPy oracle, and tiled FP32 online-softmax recurrence.
- [x] Add shape/dtype/contiguity/finite-input checks and causal/metamorphic tests, including future-K/V invariance, query locality, constant V, zero-logit means, batch/head isolation, and stability cases.
- [x] Add ordinary-CUDA baselines and fused implementations separate from cuTile: unfused CuPy, scalar CUDA, block-row CUDA, warp-row CUDA, and shared-memory tiled CUDA.
- [x] Preserve the shared-tile optimization ladder as separate versioned files (v1–v5) instead of overwriting experimental implementations.
- [x] Fix the invalid `tile_k=2` affine-V test input; the implementation was correct to reject that unsupported tile size.
- [x] Verify native CUDA correctness on A100: [run #44 — 133 passed, 1 skipped](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37756646415).
- [x] Verify broad non-cuTile regression: [run #51 — 905 passed, 52 skipped, 0 failed](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761576578).
- [x] Verify CPU/reference suite on the latest documented commit: [run #290 — 761 passed, 68 skipped](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37885170761). GPU-only skips are expected on CPU CI.
- [x] Verify isolated shared-tile v4 correctness: [30 passed](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760982099); retain as an experiment, not the default.
- [x] Verify isolated shared-tile v5 correctness: [30 passed](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761585734); retain as an experiment, not a claimed speedup.
- [x] Add warmed CUDA-event timing harnesses with 10 warmups, 30 measured iterations, medians/P90, and separate cold/JIT timing in the tile ladder.
- [x] Preserve benchmark outputs as GitHub Actions artifacts and document the run-to-run timing variability; do not use the current numbers as polished headline claims.
- [x] Capture PTXAS resource summaries for v1/v2/v3/current/v4: all showed zero reported spills; v4's resource/cubin summary matched current.
- [x] Document the R550/cuTile runtime gate, failed compatibility experiments, and Nsight Compute permission failure without presenting either as a successful workaround.
- [x] Create the consolidated [experiment report](EXPERIMENT_REPORT.md), [documentation index](README.md), [progress log](PROGRESS.md), [test matrix](TEST_MATRIX.md), [variant notes](CUDA_VARIANTS.md), and [result status snapshot](../results/STATUS.md).

## 2. Priority 0 — unblock actual cuTile

- [ ] **BLOCKED — obtain a supported execution environment.** Use an approved GPU node with a driver/runtime meeting the installed cuTile release's requirements (the RTX 4060/newer-driver path is the planned alternative) or an approved R580+ A100 path. Do not count the existing R550 runner as satisfying this requirement.
- [ ] Record the actual target GPU, driver, CUDA toolkit/runtime, Python, `cuda-tile` package, and relevant environment versions in the run log.
- [ ] Install/import the supported NVIDIA cuTile package in that environment and run a minimal compile + launch smoke test against the actual device. Save the complete logs.
- [ ] Resolve any real cuTile API/compiler/indexing issues in `src/cutile_attention.py`; keep all cuTile-specific failures distinct from ordinary-CUDA results.
- [ ] Run the actual cuTile correctness contract for D=32/64/128, all seven representative cases where supported, at least two tile configurations, and the required numerical/metamorphic checks.
- [ ] Verify the required CLI data path (`run_attention.py --input ... --output ... --backend cutile`) end-to-end and check output shape, dtype, finiteness, and reference tolerance.
- [ ] Confirm that the actual cuTile tests ran on the supported GPU instead of being skipped. Attach workflow/run URLs and test counts to the report.

## 3. Priority 1 — profiling evidence

- [ ] **BLOCKED — obtain authorized performance-counter access** on a suitable runner, or use a runner already configured to allow Nsight Compute counters. Current `ERR_NVGPUCTRPERM` runs captured no kernels; do not claim a profiler result or attempt privilege bypasses.
- [ ] Run Nsight Compute on at least two representative actual-cuTile configurations (prefer a medium case and a long/batched case); preserve `.ncu-rep`, command lines, logs, and relevant metrics.
- [ ] Inspect occupancy/resource use, register pressure, shared-memory usage, memory throughput, and any stalls the available counters support; tie optimization ideas to observed evidence.
- [ ] (Recommended) Capture an Nsight Systems timeline for the end-to-end invocation to distinguish launch/JIT/host overhead from kernel runtime.

## 4. Priority 1 — reliable performance comparison

- [ ] Run the warmed benchmark on the supported GPU: at least 10 warmups and 30 timed CUDA-event iterations per configuration, reporting median/P90 and output throughput.
- [ ] Measure cold compilation/JIT separately from steady-state kernel time.
- [ ] Compare actual cuTile against the unfused CuPy baseline and the best validated ordinary-CUDA surrogate; include optional PyTorch SDPA only when installed and validated.
- [ ] Record speedup per case/configuration, exact shape/tile parameters, GPU/toolchain metadata, and correctness status alongside timings.
- [ ] Repeat the benchmark in multiple sessions with comparable GPU load and alternating/interleaved variant order; inspect variance before promoting any claimed win.
- [ ] Save raw JSON/CSV and logs under the existing results/artifact convention without replacing earlier runs.

## 5. Priority 2 — optimization experiments, only after a valid baseline

- [ ] Add v5 to the PTXAS resource report and record its register/shared-memory/spill/cubin summary.
- [ ] Re-run current-vs-v5 with controlled, alternating benchmark order and multiple repetitions across the representative shapes and tile sizes. Promote v5 only if the gain is repeatable and no important regression appears.
- [ ] Re-measure the current shared-tile surrogate under controlled conditions if new tuning decisions depend on its absolute latencies.
- [ ] Use actual profiler evidence to choose any next optimization; preserve every materially different implementation as a new version and record both positive and negative results.
- [ ] For every kernel change, run targeted correctness first, then the broad regression suite before accepting the change.

## 6. Final deliverable and reproducibility

- [ ] Reconcile all report/status references to the latest verified workflow runs and exact tested commit SHAs.
- [ ] Update [the experiment report](EXPERIMENT_REPORT.md), [test matrix](TEST_MATRIX.md), [progress log](PROGRESS.md), [result snapshot](../results/STATUS.md), and root README with final actual-cuTile correctness, profiling, and benchmark evidence.
- [ ] Ensure the final report clearly separates actual NVIDIA cuTile results from ordinary-CUDA surrogate results and explicitly states remaining limitations.
- [ ] Verify all required course-scope deliverables are present: implementation, correctness tests, at least two tile configurations, performance comparison, separate JIT timing, and profiler evidence for at least two configurations.
- [ ] Re-run the documented test/benchmark commands from a clean checkout or reproduce them in CI; preserve the final run URLs and raw artifacts.
- [ ] Mark this project complete only when the required actual-cuTile path, correctness, benchmarking, and profiling gates above have evidence.

## Decision rules

1. Correctness is a hard gate; never promote a faster-looking kernel that fails its independent references or regression tests.
2. One noisy timing run is not sufficient to claim a general speedup.
3. A passing ordinary-CUDA test or benchmark does not count as actual cuTile execution.
4. Do not spend time retrying already-rejected compatibility hacks on R550 unless a materially new, supported runtime path becomes available.
5. After each work session, update the checkboxes and append dated details to [the progress log](PROGRESS.md). Keep this checklist and the links in the root README synchronized.
