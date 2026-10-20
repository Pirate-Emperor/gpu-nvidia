# Project TODO Checklist — Fused Causal Attention with NVIDIA cuTile

**Last reviewed:** 2026-10-20  
**Start here:** use the [Project Two-Pager](PROJECT_TWO_PAGER.md) for the current concise state and gates. This checklist explains the goal, historical evidence, remaining requirements and the evidence needed before the current source tree can be marked complete. The [root README](../README.md) is the entry point; the [experiment report](EXPERIMENT_REPORT.md) contains the detailed measurements and history.

**Authoritative experiment record:** [root Report.md](../Report.md) contains the full benchmark matrix, profiler metrics/figures, all project/CUDA/library errors, external references and optional-objective tracker. [RUN_EVIDENCE.md](../RUN_EVIDENCE.md) keeps automation-only provenance separate as requested.

**Rubric closure addendum:** use the new [course project completion checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md) and [detailed report](COURSE_PROJECT_REPORT_2026-10-19.md); those documents distinguish completed GPU gates from remaining assignment-level milestones.

**Status key:** `[x]` done and backed by evidence · `[ ]` still to do · **BLOCKED** needs a suitable environment or permission. A task is not complete merely because code exists.

## 1. Project goal, in plain English

Implement a **fused causal self-attention forward kernel**. Given FP16 query, key, and value tensors (Q, K, V), it must compute:

1. query–key dot products, accumulated in FP32;
2. scaling by 1 / sqrt(D) and the causal mask (a token can attend only to itself and earlier tokens);
3. numerically stable softmax, preferably using an online/tiled recurrence;
4. the probability-weighted sum of V, returned in FP16.

Required head dimensions are D = 32, 64, and 128. The fused kernel must not write the complete sequence-by-sequence score or probability matrix to global memory. That avoids a large intermediate allocation and is a central part of the exercise.

### What is NVIDIA cuTile?

**NVIDIA CUDA Tile (cuTile Python)** is a GPU-kernel programming model in which we express computations on multidimensional arrays and tiles—small blocks of data—instead of manually assigning each scalar operation to individual CUDA threads. The cuTile compiler/runtime maps those tile operations onto the GPU. The aim is to simplify writing tiled GPU algorithms while allowing the compiler to use hardware mechanisms such as Tensor Cores when appropriate. See NVIDIA's [cuTile Python documentation](https://docs.nvidia.com/cuda/cutile-python/) and [official repository](https://github.com/NVIDIA/cutile-python).

For this project, the important distinction is:

- **Actual cuTile target:** `src/cutile_attention.py`. The fused causal-attention kernel passes the RTX 4060 correctness matrix and controlled benchmark; two actual-cuTile Nsight Compute profiles have now been captured and inspected.
- **Ordinary-CUDA controls:** `src/custom_cuda_attention.py`, `src/warp_cuda_attention.py`, `src/shared_tile_cuda_attention.py`, and `src/variants/shared_tile_v1.py` through `shared_tile_v5.py`. These are CUDA C++/NVCC implementations that helped us validate the mathematics and compare tile strategies on the available A100.
- **Why the distinction matters:** the ordinary-CUDA tests and speed measurements are useful engineering evidence, but they are **not actual cuTile execution**, and cannot alone satisfy the course target.

The current A100 runner uses driver R550.90.07, below NVIDIA's documented R580+ cuTile Python runtime minimum. Prior CUDA-compatibility experiments did not produce a verified cuTile run. **Decision: stop using this R550 A100 as the actual-cuTile route.** Retain it for ordinary-CUDA surrogate tests. Move actual-cuTile bring-up to the user's local GPU after collecting its real GPU/OS/driver/toolchain inventory, or to another explicitly authorized supported environment. Do not assume an RTX 4060 or change a system driver without local owner/admin action. See NVIDIA's [cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html).

## 2. Current status at a glance

**Current-main boundary (20 October 2026):** the strongest native RTX 4060 results below belong to an earlier validated core snapshot. Main now includes GQA/MQA, packed-variable-length, PyTorch bridge, autotuner and load-latency-hint additions, with new compile-time parameters in the core kernel. The PowerShell startup issue was fixed in [the SDPA workflow](../.github/workflows/torch-sdpa-preflight.yml) and [the NCU workflow](../.github/workflows/cutile-rtx4060-ncu.yml); CPU/reference CI passed, but fresh native regression and profiler jobs were queued at the last check. See the [execution evidence index](EXECUTION_EVIDENCE_INDEX.md). Treat the pass rows below as historical evidence until the current-main GPU job finishes.

**Latest full native Windows validation:** [run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223), commit `cd1ec63b210682613ddb29dca3543f0cacc9bacf`; smoke, 48 required attention tests, 29 numerical configurations, CLI/oracle verification, and all three controlled benchmark sessions passed. That run also found Nsight Compute 2026.3.1.0 and completed both basic captures, but its first artifact omitted the compressed `.ncu-repz` files because the upload expected `.ncu-rep`. The dedicated follow-up [profile run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066) reused those reports, captured scheduler/warp-state reports, exported details/raw CSVs, and preserved all four reports in [artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039). The controlled benchmark snapshot and raw three-session data remain linked in [the result report](CUTILE_RTX4060_RESULTS_2026-10-19.md) and [artifact 12148414465](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910/artifacts/12148414465).

| Area | Status | Meaning |
|---|---|---|
| Mathematics and independent references | **DONE** | Independent scalar and FP32 matrix references are used for checks. |
| Ordinary-CUDA correctness on A100 | **DONE as control only** | Historical ordinary-CUDA runs do not count as cuTile results. |
| Shared-tile experiments v1–v5 | **DONE for implementation and targeted checks** | Versions are preserved; no unproven win is promoted. |
| Actual cuTile smoke and attention correctness | **Historical baseline PASS; current main pending** | The earlier RTX 4060 run passed 48 required tests with REQUIRE_CUTILE=1. Current main needs a fresh full GPU regression. |
| Numerical diagnostics | **Historical baseline PASS** | 29 legal case/tile combinations; zero mixed-tolerance violations in the earlier snapshot. |
| CLI and independent oracle | **Historical baseline PASS** | FP16 output verified by the independent NumPy oracle in the earlier snapshot. |
| Actual cuTile benchmark | **Historical baseline PASS WITH CAVEATS** | Three sessions covered five tiles; long cases were stable, very short cases noisy. |
| Nsight Compute profiles | **Historical baseline PASS; refresh pending** | Two basic reports plus two Scheduler/Warp State reports were captured and inspected; [artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039). Reprofile current source after the native regression. |
| Assignment / final course deliverable | **Historical baseline PASS; current source pending revalidation; rubric closure PARTIAL** | The earlier snapshot passed core correctness, CLI, controlled benchmark and four-profile NCU gates. Current-source revalidation, framework/vendor reference, clean-cache JIT for long cases, a second profiler-guided improvement and owner-confirmed contributions remain in [the rubric checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md).

## 3. Done so far — brief explanations and evidence

- [x] **Defined and frozen the required scope.** Causal attention forward only; FP16 Q/K/V inputs; FP32 accumulation; D=32/64/128; stable softmax; no dropout, backward pass, GQA/MQA, KV cache, or full global score/probability matrix.
- [x] **Built independent mathematical references.** A scalar FP64 oracle, a full-matrix FP32 NumPy oracle, and a separate tiled online-softmax reference provide independent checks. This reduces the risk of testing a kernel against a reference that repeats the same mistake.
- [x] **Added contract, edge-case and metamorphic tests.** Tests cover shapes/dtypes/contiguity/finite inputs, tile-size validation, causal future-K/V invariance, query locality, constant V, zero-logit prefix means, batch/head isolation, stability and long sequences.
- [x] **Implemented non-cuTile comparison paths.** The repository contains an unfused CuPy baseline, a simple scalar CUDA baseline, block-row and warp-row fused CUDA implementations, and a shared-memory tiled CUDA path. These provide correctness controls and performance reference points while cuTile is unavailable.
- [x] **Preserved the optimization experiments instead of overwriting history.** Shared-tile v1 through v5 remain separate files: a simple two-pass version, warp-row online softmax, a four-score loop, a persistent-Q-register experiment, and a `half2` QK experiment.
- [x] **Fixed a test-input issue.** An affine-V test used unsupported `tile_k=2`; it was changed to a legal tile size. The failure was in the test setup, not evidence of a kernel defect.
- [x] **Validated the native CUDA path on the A100.** [Run #44](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37756646415): 133 passed, 1 skipped.
- [x] **Ran the broad non-cuTile regression.** [Run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082): 907 passed, 52 skipped, 0 failed. This is ordinary CUDA/reference/project-contract coverage, not actual cuTile execution.
- [x] **Ran CPU/reference validation (historical).** [Run #340](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056390931): 763 passed, 68 skipped on pre-handoff main SHA afe530bc090c779ebe5d741dcf7ae544985570d1. GPU-only skips are expected on a CPU-only hosted runner.
- [x] **Updated the runner contract for persistent owner-supplied-token mode.** [Run #75](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39147796099): 764 passed, 68 skipped, 0 failed. This is CPU/reference/structure coverage, not GPU attention execution.
- [x] **Checked v4 and v5 independently.** [v4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760982099) and [v5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761585734) each passed 30 targeted GPU tests. v4 was not promoted because its PTXAS/cubin summary matched the current version and it regressed on a longer batched case. the repeated paired [benchmark #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163) showed a modest medium-D=64 gain but mixed short cases and only ~0.3–0.6% on long/batched inputs; it remains unpromoted.
- [x] **Added a reproducible preliminary benchmark ladder.** CUDA-event timing uses warmups and 30 measured iterations, reporting medians/P90; the tile ladder measures first compilation separately and preserves artifacts. The ladder shows the later tiled variants generally improve many short/medium cases over v1, but timing variability across runs is high; the measurements are preliminary, not final performance claims.
- [x] **Captured compiler resource summaries for v1–v5 and current.** [PTXAS run #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) compiled all six variants for SM80 with zero reported spills. v5 uses 32 registers and 33,792 B shared memory like current, but has a 15,520 B cubin vs 15,136 B for current.
- [x] **Recorded environment and profiling evidence honestly.** The historical A100 attempt failed with `ERR_NVGPUCTRPERM`; the later native RTX 4060 setup located Nsight Compute 2026.3.1.0 and successfully captured/inspected two actual-cuTile configurations. Current reports and the remaining stall-sampling limitation are documented in [the profile analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md).
- [x] **Prepared the supported-driver actual-cuTile workflow.** [`.github/workflows/a100-nsight.yml`](../.github/workflows/a100-nsight.yml) now runs actual cuTile correctness tests before collecting Nsight Compute profiles for both a medium and a long/batched configuration. It remains unexecuted successfully on this R550 node because its R580+ gate correctly blocks it.
- [x] **Organized the project documentation.** Root README, this checklist, experiment report, testing contract, test matrix, variant notes, environment/admin notes, progress log and result snapshot link together. Start from the [documentation index](README.md).
- [x] **Added regression checks for project tracking and the future cuTile profiling workflow.** The CPU suite now verifies key documentation paths and that the supported-driver profiling workflow gates two profiles on actual-cuTile correctness. [historical CPU run #340](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056390931): 763 passed, 68 skipped; [A100 full run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082): 907 passed, 52 skipped.

For detailed measurements, configurations, limitations and run links, read [the experiment report](EXPERIMENT_REPORT.md). The [test matrix](TEST_MATRIX.md) lists coverage and status by test family.

## 4. Next work — follow this order

### Priority 0A — supported native Windows runner validated; Docker isolation remains optional

- [x] **Inventoried the actual local machine and passed the genuine cuTile smoke gate.** RTX 4060 / CC 8.9 / driver 616.92 / Windows x64 build 26300; CPython 3.12.10 and cuTile 1.6.0 are installed in `.venv-cutile-windows`. The vector-add kernel reports `GPU_SYNCHRONIZE=PASS`, `OUTPUT_CHECK=PASS`, and `CUTILE_SMOKE=PASS`. The native Windows self-hosted Actions runner has now executed the successful actual cuTile workflow; Docker and WSL2 were not needed. See the [current results](CUTILE_RTX4060_RESULTS_2026-10-19.md).
- [x] **Design a constrained Docker runner boundary in repository code.** The constrained Docker runner remains optional; the successful current experiment ran natively on Windows and did not use the Docker runner. Do not add Docker installation to the remaining project steps.
- [ ] **Do not expose host capabilities.** Never run a privileged container, mount the host Docker socket, mount the user's home/SSH directory/browser profiles/cloud credentials, or grant sudo/administrator privileges to the job account. Restrict GPU access to compute needs and keep inbound remote access disabled.
- [x] **Define GitHub execution restrictions in the new workflow/launcher.** Repository-scoped persistent runner that the owner removes manually; the script accepts a locally supplied registration token and does not auto-dispatch. The workflow is main-only, uses contents: read, and confines package installs and caches to the project venv/root. The GPU workflow execution is confirmed by run #39160827910.
- [ ] **Keep the runner's required GitHub authentication local.** GitHub Actions runner registration/connection authentication cannot be eliminated, but no personal PAT, SSH key, or admin credentials should be available to job code. Never paste runner registration tokens, config credentials, or secrets into ChatGPT or commit them.
- [ ] **Test the boundary deliberately.** From a test job verify nvidia-smi and the CUDA compiler/runtime work, only the intended workspace is writable, outside host paths and credentials are not mounted, sudo is unavailable, and the job cannot access the host container daemon.
- [ ] **Record the exact runner boundary before using it for experiments.** Document host OS/WSL status, GPU model, driver, CUDA toolchain, isolation model, mounts, account/UID and egress policy in the handoff/status files.

Note: a self-hosted runner executes repository workflow code on the machine; it does not give ChatGPT a live terminal by itself. A container alone is not a strong boundary against every kernel/container escape, so use a dedicated VM/isolated machine when the risk level warrants it.

### Priority 0 — get actual cuTile running

- [x] **Validate the native Windows cuTile stack with a real kernel.** `scripts/cutile_smoke.py` passed on the RTX 4060 with Python 3.12.10, cuTile 1.6.0, and matched CUDA 13.4 TileIR compiler packages. The follow-on actual attention suite also passed (run #39160827910). Keep the existing R550 A100 for ordinary-CUDA surrogate validation only.
- [x] **Record the environment before changing code.** Save GPU model/compute capability, driver, CUDA toolkit/runtime, Python, cuTile package version, and exact commands to the progress log or a dated run record.
- [x] **Run a minimal actual-cuTile smoke test.** The real kernel compiled, launched, synchronized, and matched the expected output for all 4,096 elements. Preserve the console log; this separates environment readiness from attention-kernel bugs.
- [x] **Validate `src/cutile_attention.py` on a supported GPU.** Run #39160827910 passed the frozen attention matrix after the causal K/V bound and FP16 P×V optimizations.
- [x] **Run the actual-cuTile correctness matrix.** Cover D=32/64/128, the seven representative short/medium/long cases where supported, at least two legal tile configurations, mixed-precision tolerances, numerical stability, causal masking, future-K/V invariance, and input validation.
- [x] **Verify the command-line path end to end.** Run `python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile`; check the output file, shape, FP16 dtype, finite values and tolerance against an independent oracle.
- [x] **Record proof that the actual tests executed.** Save the run URL, tested commit, GPU/toolchain versions, pass/skip/fail counts, and logs. A workflow that skips cuTile tests does not count.

**Historical Priority 0 baseline passed:** the earlier kernel compiled/launched and passed the required correctness suite (run #39160827910). Because the current main adds compile-time kernel parameters and optional launch paths, rerun that full gate before marking the current source revalidated.

### Priority 1 — obtain profiler evidence

- [x] **Make Nsight Compute available and capture actual-cuTile reports.** The official Windows Nsight Compute CLI 2026.3.1.0 is installed; counters worked on the RTX 4060 without `ERR_NVGPUCTRPERM`. [Profile run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066) preserves the reports and exports.
- [x] Capture two actual cuTile configurations: `medium_d128` at 64×64 and `longer_batched_d128` at 64×128. Both compressed `.ncu-repz` reports and their command logs are preserved in [artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).
- [x] Inspect register/shared-memory usage, occupancy, throughput, scheduler eligibility, and execution-pipeline-stall diagnostics. Findings and limitations are recorded in [CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md).
- [x] Capture Nsight Systems timelines and CUDA API/kernel summaries for the warmed ordinary-CUDA surrogate at `medium_d128, tile_k=64` and `longer_batched_d128, tile_k=32`: [run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950), [uploaded trace + summary artifacts](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954). This is supplementary surrogate evidence; actual-cuTile timelines remain to be captured after cuTile runs on supported hardware.

**Historical profiler gate complete:** two successful, inspectable actual-cuTile reports were captured and interpreted. The current source should be re-profiled after the fresh regression; scheduler/warp-state analysis remains directional because `smsp__pcsamp_sample_count` was unavailable. Do not overinterpret Nsight's estimated local-speedup heuristics.

### Priority 1 — complete a reliable performance comparison

- [x] Benchmark actual cuTile after correctness passed. Three sessions use 10 warmups and 30 CUDA-event samples with five launches averaged per sample; median/P90, first-call, throughput, speedup, and raw JSON/CSV are preserved.
- [x] Record first-call/JIT-associated wall time separately from steady-state event timing, with the caveat that a compiler-cache hit can be included.
- [x] Compare actual cuTile with the unfused FP32 CuPy baseline. [ ] A CUDA PyTorch SDPA comparison was not run because a CUDA-enabled build is not installed in the project venv.
- [x] Report per-case latency/speedup, tiles, environment, correctness, and the raw artifact in CUTILE_RTX4060_RESULTS_2026-10-19.md.
- [x] Run three sessions with alternate tile order and report spread and baseline drift; short noisy timings are excluded from headline claims.
- [x] Archive raw three-session JSON/CSV and summary plus logs in artifact 12148414465; earlier artifacts remain preserved.

### Priority 2 — optimize only after a correct, measured actual-cuTile baseline

- [x] Add v5 to the PTXAS resource report and record registers, shared-memory use, spills and cubin size; see [run #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187).
- [x] Retest current-vs-v5 with three paired sessions and alternating measurement order across all seven cases and tile_k=16/32/64: [run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163). Result: v5 is not promoted; gains are mixed/noisy on short cases and about 0.3–0.6% on long/batched cases, although medium D=64 improves modestly.
- [ ] Re-measure the current ordinary-CUDA surrogate under controlled conditions when a decision depends on its absolute latency.
- [x] Use actual-cuTile profiler evidence to identify the next optimization target: shared-memory/register pressure and instruction scheduling. Keep the validated 64×64 default and 64×128 longer-batched specialization; the profile alone does not prove an untested change will win.
- [ ] For each change, run targeted correctness first, then the broad regression suite before accepting the change.

## 5. Final submission and definition of done

- [x] Update the root README, experiment report, test matrix, progress log, and results snapshot with current evidence.
- [x] Verify core deliverables: fused cuTile implementation, independent correctness tests, multiple tiles, unfused FP32 CuPy comparison, at least 30 measured iterations, separate first-call/JIT-associated timing, and two actual-cuTile profiler configurations. A CUDA-enabled PyTorch SDPA comparison was not available in the existing project venv and remains optional follow-up; current speedups are explicitly against the unfused CuPy baseline.
- [x] Reproduced through native Windows workflow checkout and preserved benchmark output in artifact 12148414465 and profiler output in [artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).
- [x] Keep ordinary-CUDA metrics labelled as controls; actual cuTile data are tabulated separately.
- [ ] Re-run and inspect the full native correctness, numerical, CLI, benchmark and NCU gates against current main after the optional additions; then reaffirm current-tree closure or document any regressions.

## Decision rules

1. Correctness is a hard gate; speed cannot compensate for incorrect output.
2. One noisy benchmark run does not prove a general speedup.
3. Ordinary-CUDA success does not count as actual cuTile success.
4. Do not spend more time on failed R550 compatibility hacks without a materially new supported path.
5. After each work session, update this checklist and append dated evidence to [the progress log](PROGRESS.md).
