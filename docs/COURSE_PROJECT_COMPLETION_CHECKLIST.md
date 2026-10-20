# Course Project Completion Checklist — Fused Attention with cuTile

**Source of requirements:** nvidia2-cutile.pdf, IIT Madras GPU Programming course project proposal (six pages).  
**Reconciled:** 19 October 2026.  
**Status:** [x] PASS means verified; [~] PARTIAL means a stated rubric detail remains open; [ ] OPEN means not demonstrated; [O] OPTIONAL means outside required scope.

## Executive status

The fused cuTile kernel and selected optional GPU features have a fresh complete native pass. The assignment should not yet be described as fully closed. Outstanding rubric-level items are: framework/vendor attention comparison (currently blocked by a plain PyTorch CUDA GEMM failure on the pinned CUDA 13.0 wheel), matched NCU validation of the medium-workload LOAD_LATENCY=10 timing candidate, clean-cache JIT timing for long specializations, and the owner's truthful individual-contributions statement. A CUDA 13.2 PyTorch build is being checked in a separate project-local venv; the existing cuTile venv remains unchanged.

## Current validation evidence

- [x] Latest complete native actual-cuTile validation: [workflow #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), commit bf5a6233e195c2ced647b4e20b5b2d914da60e85. It passes the core and selected optional GPU correctness gates, 29-case numerical matrix, CLI oracle, three-session benchmark, all accuracy-gated hint trials, and two NCU captures. Artifact: [#12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195).
- [x] Latest CPU/reference CI after doc updates and temporary-file cleanup: [workflow #39231291000](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39231291000) — 764 passed, 68 skipped, zero failed. GPU modules skip in CPU CI because CuPy is absent; this does not replace GPU evidence.
- [x] Dedicated profiler export: [workflow #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066), [artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).
- [x] Four compressed .ncu-repz reports, four Details exports and four Raw CSV exports downloaded and inspected. No ERR_NVGPUCTRPERM on RTX 4060.
- [x] Temporary PDF export workflow deleted after retrieving the original problem PDF; read-back confirms it is absent from current main.

## Gate A — implementation and interface

- [x] Genuine NVIDIA cuTile implementation in src/cutile_attention.py; ordinary CUDA surrogates do not substitute for this result.
- [x] Causal scaled dot-product self-attention forward path.
- [x] FP16 Q/K/V inputs; FP32 QK accumulation, online-softmax state, row sums and P×V accumulation; output FP16.
- [x] Required head dimensions D=32, 64, 128 are exercised in the legal workload/tile matrix.
- [x] Score, scaling, causal mask, softmax and P×V execute in one fused cuTile kernel launch per output call.
- [x] No full N×N score/probability tensor is materialized globally in the fused path.
- [x] Stable online-softmax recurrence explained in the technical report.
- [x] Five legal tile shapes exercised where supported: 32×64, 64×64, 64×128, 128×64, 128×128.
- [x] Input-contract tests cover supported shapes/types and tile divisibility. Keep these tests after every material code change.
- [x] CLI contract works: python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile.
- [x] Deterministic CLI output passed the independent NumPy oracle; output shape/dtype/finiteness passed.

## Gate B — correctness and numerical stability

- [x] Real cuTile smoke kernel compiled/launched/synchronized and verified 4,096 outputs exactly.
- [x] Required attention suite with REQUIRE_CUTILE=1: 48 passed, 0 failed.
- [x] Independent FP32 NumPy diagnostics: 29 legal case/tile configurations, zero tolerance violations.
- [x] Fixed seeds and finite-output checks are recorded in the suite and diagnostics.
- [x] Mixed tolerance: abs(error) <= 0.02 + 0.02 * abs(reference).
- [x] Maximum absolute error 0.0009765625; largest per-configuration P99 absolute error 0.00048828125; largest RMSE 0.000086860741.
- [x] Maximum relative error where abs(reference) >= 0.01: 0.02245863 (2.25%); max tolerance ratio 0.024366 (about 2.44% of the allowed envelope).
- [x] Preserve numerical JSON/logs; do not describe the 2.44% tolerance-budget ratio as 2.44% output error.

## Gate C — performance evaluation

- [x] Deliberately unfused FP32 CuPy baseline that materializes score/probability intermediates.
- [x] Seven workloads and 29 legal workload/tile pairs across five tile candidates where legal.
- [x] Ten warmups and 30 CUDA-event samples per configuration; each sample averages five launches (150 timed launches per case/tile/session).
- [x] Three independent sessions with rotated/reversed tile order.
- [x] Baseline measured before and after each sweep; drift reported.
- [x] Median, P90, speedup, session range and output-byte throughput recorded with raw JSON/CSV.
- [x] Short/medium/long workloads include D=64 and D=128.
- [~] JIT/first-call time is reported when captured, but long and long-batched specializations have null JIT fields because they were already compiled. A deliberate clean-cache measurement or a documented exception is needed to fully close this sub-requirement.
- [ ] **Framework/vendor attention reference blocked before timing.** PyTorch 2.14.0+cu130 reports CUDA available, but a plain float32 torch.bmm smoke fails with CUBLAS_STATUS_INVALID_VALUE before the cuTile bridge runs: [diagnostic run #39423190840](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840). A separate CUDA 13.2 wheel probe is queued in [run #39423804493](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423804493); no backend or PyTorch speed claim is made. Current speedups remain vs unfused FP32 CuPy.
- [x] Tile recommendation reflects evidence: keep 64×64 as general default; 64×128 is promising for long-batched D=128. On long D=128, 64×64 and 64×128 are effectively tied (0.273408 vs 0.273203 ms).
- [x] Short-case timing caveat is explicit: fused-median range reaches about 197%; maximum baseline drift reaches 81.17%; do not headline noisy short-case ratios.

## Gate D — Nsight Compute profiling

- [x] Official Nsight Compute CLI 2026.3.1.0 located at the standard NVIDIA installation path; version recorded.
- [x] medium_d128, tile 64×64: basic throughput/resource report plus SchedulerStats/WarpStateStats report.
- [x] longer_batched_d128, tile 64×128: basic throughput/resource report plus SchedulerStats/WarpStateStats report.
- [x] Four .ncu-repz reports plus four Details exports and four Raw CSV files preserved in [artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).
- [x] Registers, shared memory, occupancy, SM/DRAM/L1/L2 throughput and scheduler eligibility inspected.
- [x] Counter access succeeded on RTX 4060; no ERR_NVGPUCTRPERM. Earlier A100 permissions error is separate historical evidence.
- [x] Limit recorded: smsp__pcsamp_sample_count unavailable; no detailed source-level stall sampling. NCU estimated local-speedup outputs are heuristics, not measured wins.
- [x] Seven native Nsight Compute GUI screenshots are preserved with the actual report files and exports; one longer-batched LOAD_LATENCY=4 screenshot is explicitly skipped in the capture manifest after GetWindowRect failed. The matched medium LOAD_LATENCY=10 profile is queued for the current hint decision.

## Gate E — four-week milestone tracking

- [x] **Week 1: correct fused kernel.** References, baseline, kernel and deterministic tests have actual-cuTile validation.
- [~] **Week 2: initial profiling and optimization.** Timing now identifies medium D=128 / tile 64×64 / LOAD_LATENCY=10 as a credible candidate: 0.053248 ms vs 0.058342 ms compiler default (8.73% gain) with about 0.53% session range for both. Matched NCU review remains open; global default stays 0.
- [ ] **Week 3: second profiler-guided improvement.** Choose a hypothesis from shared-memory/register pressure or scheduling signals; record exact code diff; run targeted correctness first, full correctness, alternating-order benchmark repeats and before/after NCU exports. Record failed experiments too.
- [~] **Week 4: final report/robustness.** Final GPU correctness/benchmark/profile evidence exists and a seven-page report is prepared. Close remaining reference, JIT, contributions and optimization items or document approved exceptions.

## Gate F — report and repository deliverables

- [x] README/setup, CLI, kernel source, independent references, unfused baseline, automated tests, benchmark scripts and raw JSON/CSV are present.
- [x] Detailed results and profiler-analysis Markdown are present in docs.
- [x] Technical report explains algorithm/work decomposition/tensor layouts, online-softmax derivation, correctness, performance plots/method, profiler results, optimization history, limitations and artifacts.
- [x] Replacement report targets seven pages within the assignment’s 5–8 page range; rendered pages must be visually checked before submission.
- [ ] Confirm author/team names and truthful individual contributions; names/responsibilities are not guessed.
- [x] Temporary PDF-export workflow removed; latest post-cleanup CPU/reference CI passed.

## Gate G — optional extension backlog

Do not start optional work before the above rubric gaps are closed or accepted. Seek approval before installing packages or making system-level changes.

- [ ] **PyTorch-model integration:** integrate the kernel into an example model, verify outputs, and show measurable end-to-end improvement over that model’s baseline attention.
- [ ] **BF16/FP8:** update contract, reference, numerical tests, benchmark and profile; this changes the numerical scope.
- [ ] **Backward propagation:** derive gradients, compare to autograd and test edge cases; retain separate from the forward-only kernel.
- [ ] **Variable-length/packed input:** define representation/masking contract and correctness oracle.
- [ ] **GQA/MQA:** support distinct query/KV head counts with layout, broadcast, correctness and performance tests.
- [ ] **Autotuning:** reproducible selection across legal tile shapes; do not promote choices from noisy short timings.
- [ ] **CuTe DSL comparison:** this is an instructor-approved alternative track, not required in addition to cuTile. Proceed only with instructor approval or explicit scope decision.
- [ ] **Architecture-specific pipelining/async data movement:** hypothesis-driven experiment, correctness first, then controlled repeat benchmarks and profiling.

## Recommended execution order

1. Confirm individual contributions and finalize the seven-page report.
2. With owner approval, benchmark a compatible CUDA-enabled framework/vendor attention reference and document which backend ran.
3. Run clean-cache JIT measurement for long cases, or clearly document why it cannot be made reliable.
4. Conduct one profiler-guided candidate experiment: targeted correctness → full correctness → alternating-order benchmark repeats → two-profile comparison.
5. Reconcile checklist and report; run CPU/reference CI after documentation changes and native GPU workflow only after kernel/code changes.
6. Only then schedule optional extensions.

## Evidence index

- Full native validation: [run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223)
- Core run artifact: [#12149980979](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979)
- Post-cleanup CPU/reference CI: [run #39231291000](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39231291000) — 764 passed, 68 skipped, zero failed; GPU skips are expected on CPU-only CI.
- NCU run: [#39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066)
- Profile artifact: [#12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039)
- Results: [CUTILE_RTX4060_RESULTS_2026-10-19.md](CUTILE_RTX4060_RESULTS_2026-10-19.md)
- Profiler analysis: [CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md)
