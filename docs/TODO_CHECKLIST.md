# Project TODO Checklist — Fused Causal Attention with NVIDIA cuTile

**Last reviewed:** 2026-10-09  
**Start here:** this file explains the goal, what has already been done, what remains, and the evidence required before the course project can be marked complete. The [root README](../README.md) is the entry point; the [experiment report](EXPERIMENT_REPORT.md) contains the more detailed measurements and history.

**Status key:** `[x]` done and backed by evidence · `[ ]` still to do · **BLOCKED** needs a suitable environment or permission. A task is not complete merely because code exists.

## 1. Project goal, in plain English

Implement a **fused causal self-attention forward kernel**. Given FP16 query, key, and value tensors (Q, K, V), it must compute:

1. query–key dot products, accumulated in FP32;
2. scaling by (1/\sqrt{D}) and the causal mask (a token can attend only to itself and earlier tokens);
3. numerically stable softmax, preferably using an online/tiled recurrence;
4. the probability-weighted sum of V, returned in FP16.

Required head dimensions are D = 32, 64, and 128. The fused kernel must not write the complete sequence-by-sequence score or probability matrix to global memory. That avoids a large intermediate allocation and is a central part of the exercise.

### What is NVIDIA cuTile?

**NVIDIA CUDA Tile (cuTile Python)** is a GPU-kernel programming model in which we express computations on multidimensional arrays and tiles—small blocks of data—instead of manually assigning each scalar operation to individual CUDA threads. The cuTile compiler/runtime maps those tile operations onto the GPU. The aim is to simplify writing tiled GPU algorithms while allowing the compiler to use hardware mechanisms such as Tensor Cores when appropriate. See NVIDIA's [cuTile Python documentation](https://docs.nvidia.com/cuda/cutile-python/) and [official repository](https://github.com/NVIDIA/cutile-python).

For this project, the important distinction is:

- **Actual cuTile target:** `src/cutile_attention.py`. This is the required implementation path, but it has **not yet been validated on a supported GPU/runtime**.
- **Ordinary-CUDA controls:** `src/custom_cuda_attention.py`, `src/warp_cuda_attention.py`, `src/shared_tile_cuda_attention.py`, and `src/variants/shared_tile_v1.py` through `shared_tile_v5.py`. These are CUDA C++/NVCC implementations that helped us validate the mathematics and compare tile strategies on the available A100.
- **Why the distinction matters:** the ordinary-CUDA tests and speed measurements are useful engineering evidence, but they are **not actual cuTile execution**, and cannot alone satisfy the course target.

The current A100 runner uses driver R550.90.07, while the project's cuTile environment check requires R580+. Prior CUDA-compatibility experiments did not produce a verified cuTile run. The planned alternative is an approved GPU with a supported, newer driver (for example, the intended RTX 4060 path), or another approved environment meeting the installed cuTile version's requirements. Do not assume the hardware/package combination works until a real compile and launch succeeds.

## 2. Current status at a glance

| Area | Status | Meaning |
|---|---|---|
| Mathematics and independent references | **DONE** | Three separate references exist to catch algorithm and numerical mistakes. |
| Ordinary-CUDA correctness on A100 | **DONE** | Base/native and broad regression workflows passed. |
| Shared-tile experiments v1–v5 | **DONE for implementation and targeted checks** | Variants are preserved; v4/v5 are not promoted as proven speedups. |
| Timing harness and preliminary performance data | **DONE for surrogate CUDA** | Timings and raw artifacts exist, but run-to-run variability means the headline comparison needs controlled repeats. |
| Actual NVIDIA cuTile compile, launch and correctness | **OPEN / BLOCKED** | No verified end-to-end cuTile GPU run on a supported environment yet. |
| Actual cuTile benchmark | **OPEN** | Cannot compare final target performance until it executes correctly. |
| Nsight Compute hardware-counter evidence | **OPEN / BLOCKED** | Existing run hit `ERR_NVGPUCTRPERM`; no kernel was profiled. |
| Final course deliverable | **NOT COMPLETE** | Actual-cuTile correctness, performance and profiler evidence remain outstanding. |

## 3. Done so far — brief explanations and evidence

- [x] **Defined and frozen the required scope.** Causal attention forward only; FP16 Q/K/V inputs; FP32 accumulation; D=32/64/128; stable softmax; no dropout, backward pass, GQA/MQA, KV cache, or full global score/probability matrix.
- [x] **Built independent mathematical references.** A scalar FP64 oracle, a full-matrix FP32 NumPy oracle, and a separate tiled online-softmax reference provide independent checks. This reduces the risk of testing a kernel against a reference that repeats the same mistake.
- [x] **Added contract, edge-case and metamorphic tests.** Tests cover shapes/dtypes/contiguity/finite inputs, tile-size validation, causal future-K/V invariance, query locality, constant V, zero-logit prefix means, batch/head isolation, stability and long sequences.
- [x] **Implemented non-cuTile comparison paths.** The repository contains an unfused CuPy baseline, a simple scalar CUDA baseline, block-row and warp-row fused CUDA implementations, and a shared-memory tiled CUDA path. These provide correctness controls and performance reference points while cuTile is unavailable.
- [x] **Preserved the optimization experiments instead of overwriting history.** Shared-tile v1 through v5 remain separate files: a simple two-pass version, warp-row online softmax, a four-score loop, a persistent-Q-register experiment, and a `half2` QK experiment.
- [x] **Fixed a test-input issue.** An affine-V test used unsupported `tile_k=2`; it was changed to a legal tile size. The failure was in the test setup, not evidence of a kernel defect.
- [x] **Validated the native CUDA path on the A100.** [Run #44](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37756646415): 133 passed, 1 skipped.
- [x] **Ran the broad non-cuTile regression.** [Run #51](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761576578): 905 passed, 52 skipped, 0 failed.
- [x] **Ran CPU/reference validation.** [Run #290](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37885170761): 761 passed, 68 skipped. GPU-only skips are expected on a CPU-only hosted runner.
- [x] **Checked v4 and v5 independently.** [v4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760982099) and [v5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761585734) each passed 30 targeted GPU tests. v4 was not promoted because its PTXAS/cubin summary matched the current version and it regressed on a longer batched case. v5 showed promising results in some cases, but other timings were neutral/slower and noisy, so its speedup remains unproven.
- [x] **Added a reproducible preliminary benchmark ladder.** CUDA-event timing uses warmups and 30 measured iterations, reporting medians/P90; the tile ladder measures first compilation separately and preserves artifacts. The ladder shows the later tiled variants generally improve many short/medium cases over v1, but timing variability across runs is high; the measurements are preliminary, not final performance claims.
- [x] **Captured compiler resource summaries.** PTXAS summaries exist for v1/v2/v3/current/v4. They reported zero spills; v4's resource/cubin summary matched the current kernel. v5 is not yet included.
- [x] **Recorded environment and profiling blockers honestly.** The current driver/package gate and failed compatibility attempts are documented. Nsight Compute is installed, but the existing counter-profile attempt failed with `ERR_NVGPUCTRPERM` and captured no kernel.
- [x] **Organized the project documentation.** Root README, this checklist, experiment report, testing contract, test matrix, variant notes, environment/admin notes, progress log and result snapshot link together. Start from the [documentation index](README.md).

For detailed measurements, configurations, limitations and run links, read [the experiment report](EXPERIMENT_REPORT.md). The [test matrix](TEST_MATRIX.md) lists coverage and status by test family.

## 4. Next work — follow this order

### Priority 0 — get actual cuTile running

- [ ] **BLOCKED: establish a supported GPU environment.** Use an approved newer-driver GPU (the planned RTX 4060 path, if available and compatible) or an approved A100 environment meeting the installed cuTile requirements. Do not retry the previously rejected R550 compatibility setup unless the underlying runtime path materially changes.
- [ ] **Record the environment before changing code.** Save GPU model/compute capability, driver, CUDA toolkit/runtime, Python, cuTile package version, and exact commands to the progress log or a dated run record.
- [ ] **Run a minimal actual-cuTile smoke test.** Import cuTile, compile a tiny kernel, launch it on the real GPU, validate its output, and save the full log. This separates environment failures from attention-kernel bugs.
- [ ] **Validate and debug `src/cutile_attention.py`.** Resolve real cuTile API, tile-shape, indexing, launch/grid and compiler issues that appear. Keep failures in the cuTile path separate from the known-good ordinary-CUDA controls.
- [ ] **Run the actual-cuTile correctness matrix.** Cover D=32/64/128, the seven representative short/medium/long cases where supported, at least two legal tile configurations, mixed-precision tolerances, numerical stability, causal masking, future-K/V invariance, and input validation.
- [ ] **Verify the command-line path end to end.** Run `python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile`; check the output file, shape, FP16 dtype, finite values and tolerance against an independent oracle.
- [ ] **Record proof that the actual tests executed.** Save the run URL, tested commit, GPU/toolchain versions, pass/skip/fail counts, and logs. A workflow that skips cuTile tests does not count.

**Priority 0 is complete only when an actual cuTile kernel successfully compiles and launches on the supported GPU and the required correctness suite passes.**

### Priority 1 — obtain profiler evidence

- [ ] **BLOCKED: obtain authorized hardware-counter access.** Use a runner configured for Nsight Compute counters or ask the responsible administrator to enable the approved access. The current `ERR_NVGPUCTRPERM` run profiled no kernels.
- [ ] Profile at least two real cuTile configurations, ideally one medium shape and one long/batched shape. Preserve the `.ncu-rep` files, command line, logs and metrics.
- [ ] Inspect the metrics actually available: register/shared-memory use, occupancy, memory throughput and relevant stalls. Tie each proposed optimization to evidence rather than guessing.
- [ ] (Recommended) Capture an Nsight Systems timeline for the complete invocation to understand launch, compilation and host overhead separately from kernel time.

**Profiler completion means at least two successful, inspectable profiles of the actual cuTile kernel—not merely a working Nsight Compute executable.**

### Priority 1 — complete a reliable performance comparison

- [ ] Benchmark actual cuTile only after correctness passes. Use at least 10 warmups and 30 CUDA-event timed iterations per configuration; report median, P90 and output throughput.
- [ ] Measure cold compile/JIT separately from steady-state latency.
- [ ] Compare actual cuTile with the unfused CuPy baseline and the best validated ordinary-CUDA surrogate; include PyTorch SDPA only if installed and validated.
- [ ] Report per-case latency and speedup with exact shape, tile settings, device/driver/toolchain, correctness status and raw artifact links.
- [ ] Repeat sessions under comparable GPU load and alternate/interleave variant order. Check variation before making a performance claim.
- [ ] Save raw JSON/CSV and logs under the existing results/artifact convention; never overwrite earlier experiment evidence.

### Priority 2 — optimize only after a correct, measured actual-cuTile baseline

- [ ] Add v5 to the PTXAS resource report and record registers, shared-memory use, spills and cubin size.
- [ ] Retest current-vs-v5 with alternating order and multiple sessions on representative shapes and tile sizes. Promote it only if gains are repeatable and meaningful regressions are absent.
- [ ] Re-measure the current ordinary-CUDA surrogate under controlled conditions when a decision depends on its absolute latency.
- [ ] Use actual-cuTile profiler evidence to choose the next tile or memory optimization. Save each materially different approach as a new version and record both wins and regressions.
- [ ] For each change, run targeted correctness first, then the broad regression suite before accepting the change.

## 5. Final submission and definition of done

- [ ] Update the root README, [experiment report](EXPERIMENT_REPORT.md), [test matrix](TEST_MATRIX.md), [progress log](PROGRESS.md) and [result snapshot](../results/STATUS.md) with the final actual-cuTile results and exact run/commit references.
- [ ] Verify required deliverables: actual fused cuTile implementation, independent correctness tests, at least two tile configurations, comparison against an unfused baseline and an appropriate framework/vendor reference where available, at least 30 timed iterations, separate JIT timing, and profiler evidence for at least two configurations.
- [ ] Reproduce the documented commands from a clean checkout or CI run and preserve the raw outputs.
- [ ] Clearly label all ordinary-CUDA measurements as surrogate/control results; do not mix them with cuTile timing tables.
- [ ] Mark the project complete only after actual-cuTile correctness, performance and profiling gates have supporting evidence.

## Decision rules

1. Correctness is a hard gate; speed cannot compensate for incorrect output.
2. One noisy benchmark run does not prove a general speedup.
3. Ordinary-CUDA success does not count as actual cuTile success.
4. Do not spend more time on failed R550 compatibility hacks without a materially new supported path.
5. After each work session, update this checklist and append dated evidence to [the progress log](PROGRESS.md).
