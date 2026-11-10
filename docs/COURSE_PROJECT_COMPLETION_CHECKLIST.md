# Course project completion checklist

**Source of truth:** repository-root `nvidia2-cutile.pdf` (six pages).  
**Last audited:** 5 November 2026.  
**Detailed specification mapping:** [Assignment brief audit](ASSIGNMENT_BRIEF_AUDIT.md).

This checklist distinguishes historical evidence from proof for current `main`. A historical PASS is not a current-source PASS when the kernel or validation code changed after the run.

## 1. Kernel and input/output contract

- [x] Implement forward causal scaled dot-product attention as a genuine NVIDIA cuTile kernel.
- [x] Tile Q/K/V and fuse score calculation, scaling, causal masking, softmax and value accumulation in the attention launch.
- [x] Use FP16 input/output with FP32 score, softmax and P×V accumulation in the tested core path.
- [x] Keep the full N×N score/probability matrix out of global memory.
- [x] Maintain a numerically stable online-softmax state across K/V tiles.
- [x] Provide the NPZ input / NPY output CLI and a `--backend cutile` path.
- [x] Support every PDF-legal independent `D,Dv ∈ {32,64,128}`. The equality restriction was removed; unequal-dimension GPU oracle tests passed in current native RTX 4060 validation [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189).
- [x] Enforce and test the PDF bounds `1≤B≤16`, `1≤H≤32`, `1≤N≤8192`. Shared host validation and the direct device-launch path enforce the limits; boundary/just-outside contract tests and the current native suite pass.
- [x] Check unsupported head dimensions and query/key tile divisibility with clear errors for the currently implemented contract.

## 2. Correctness

- [x] Keep an independent FP32 NumPy reference and additional scalar/tiled reference code.
- [x] Validate the trusted framework FP32 reference: CUDA PyTorch SDPA vs NumPy passed on 7/7 distinct workloads, maximum absolute discrepancy 0.00048828125 ([workflow #41511770773](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41511770773)).
- [x] Exercise causal masking, fixed seeds, small/large values, finite outputs, invalid inputs and multiple batch/head/sequence workloads in the existing suite.
- [x] Historical native RTX 4060 run [#39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438): 48 core GPU tests passed, 29 legal case/tile numerical configurations passed, zero mixed-tolerance violations, and the CLI/oracle check passed.
- [x] Repeat the genuine cuTile smoke, expanded current 57-test GPU suite, CLI check and 29-case/tile matrix on the current implementation source. Native RTX 4060 workflow [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189) passed the expanded 57-test suite (including six additional 64×32 cases over the seven divisible workload lengths).
- [x] Extend correctness coverage to independent Dv and the full B/H/N contract boundaries. Unequal-D/Dv GPU oracle cases and host/device launch contract tests passed in the current native workflow and CPU suite.

## 3. Performance

- [x] Retain the deliberately unfused FP32 CuPy baseline.
- [x] Current-source benchmark matrix covers seven workload families and 29 legal case/tile rows, including short/long workloads and D=64/D=128. Three sessions are preserved; 13 rows meet conservative ≤5% fused-range and ≤5% baseline-drift filters; noisy rows are explicitly flagged.
- [x] Historical benchmark used warmups, CUDA events, 30 timed samples, repeated order-varied sessions, medians/P90 and output-byte throughput.
- [x] Preserve raw JSON/CSV timing evidence and state that speedups are against the unfused FP32 CuPy baseline, not optimized FlashAttention/SDPA.
- [x] Measure clean-cache first-call/first-launch time for two long specializations in fresh processes with empty cuTile/driver cache directories. First call includes compilation, launch and synchronization; see [current-run evidence](CURRENT_RUN_2026-11-02.md).
- [x] Complete a compatible framework/vendor attention reference: [run #41511770773](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41511770773) measured PyTorch 2.14 efficient SDPA (aten::_efficient_attention_forward) across 10 configs × 3 sessions; all 30 correctness checks passed. The same run compared the separate FP32 PyTorch SDPA reference with NumPy on 7/7 workload families. This is not the graded speedup denominator.
- [x] Record all available GPU/driver/runtime/DSL and compilation metadata. The benchmark captured a 2700 MHz SM-clock snapshot and 115 W power limit; power draw and a locked-clock policy were unavailable and are explicitly not claimed.

## 4. Nsight Compute and optimization

- [x] Capture 27 native `.ncu-repz` reports on the RTX 4060 / SM89; each native report passed its capture status check.
- [x] Preserve Details/Raw exports, PTX/SASS exports and 27 per-configuration metric analyses.
- [x] Save 864 primary GUI images (32 per profile) and 108 PTX-supplement images (4 per profile): 972 images total, 36 per configuration, all 1936×1048.
- [ ] Separate owner-requested add-on (not in the six-page PDF): complete literal manual transcription of every visual detail in all 972 full-resolution screenshots. All 27 contact sheets and representative full-resolution views have been checked, but this additional request is not a course-assignment acceptance item. See [manual-inspection status](ncu/rtx4060/27-profile-gui-analysis/MANUAL_INSPECTION_STATUS.md).
- [x] Corrected horizontal movement is proven for all 27 reports: 216 SASS/PTX pane checks, zero failures, and 486 baseline/scrolled/restored PNGs. Per-profile archives and integrity manifest are published in [the verified evidence index](ncu/rtx4060/horizontal-proof-2026-11-02/README.md).
- [x] Use at least two real profiles to discuss register/shared-memory use, occupancy, compute/DRAM throughput and scheduler/warp state; preserve limitations around source attribution and NCU rule heuristics.
- [x] Complete the first profiler-guided hint optimization. The expanded 5 November three-session sweep accepted `LOAD_LATENCY=10` for medium D=128 / 64×64: median 0.057344→0.050074 ms (12.68%), with a 2.25% candidate session range against an 8.04% default range. Matched NCU duration fell from 52.80 to 47.49 µs; static shared memory rose 65.54→98.30 KiB and occupancy remained ~8.3%. Keep the hint workload-specific; other workloads, including longer-batched D=128, retain hint 0.
- [x] Complete a second, distinct profiler-guided improvement loop for the longer-batched D=128 workload. On the current expanded source, 64×128 improved the stable three-session median from 0.936346 ms to 0.921395 ms (1.60%); both rows passed ≤5% spread/drift gates. Matched NCU duration fell from 997.73 µs to 990.98 µs (0.68%), with memory throughput 19.65%→20.21% and DRAM throughput 11.67%→11.81%. Trade-off: registers/thread 227→255 and static shared memory 65.54→81.92 KiB; occupancy remained 8.33%, and no-eligible cycles rose 84.48%→85.24%. Keep 64×128 workload-specific; retain global 64×64. See [Week 3 matched NCU analysis](ncu/rtx4060/WEEK3_TILE_CHOICE_2026-11-05.md), [benchmark run #42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189), and [NCU run #42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947).

## 5. Final report and submission materials

- [x] Keep implementation, references, tests, CLI, benchmark scripts and raw result artifacts in the repository.
- [x] Keep the existing technical report and per-configuration analysis files as working drafts.
- [x] Update the working milestone record against current-source RTX 4060 validation in [CURRENT_RUN_2026-11-02.md](CURRENT_RUN_2026-11-02.md). The older Report.md remains a historical working draft, not the final report.
- [x] Freeze limitations, negative results, algorithm/tile derivation, optimization history and profiler discussion in the [final submission report](SUBMISSION_REPORT_2026-11-03.md).
- [x] Record verifiable repository contribution attribution: Git history identifies xemptynothing as the human author of project changes; automated evidence commits are identified as github-actions[bot]. The report explicitly avoids inventing off-repository contributors or percentages.
- [x] Produce the final 5–8 page submission report: [Markdown](SUBMISSION_REPORT_2026-11-03.md) and [PDF](SUBMISSION_REPORT_2026-11-03.pdf) (six A4 pages).

## 6. Optional extensions (not core closure blockers)

BF16/FP8, backward propagation, grouped/multi-query attention, packed variable-length sequences, autotuning, the alternative CuTe DSL track and architecture-specific pipelining are optional in the assignment brief. They must not be presented as required core acceptance items unless the project owner has elected to include them.

## Report-stage decision

**Assignment state: COMPLETE for the six-page PDF scope.** Week 1 is verified on the RTX 4060 with the expanded 57-test suite and 29/29 numerical matrix; Week 2's first hint optimization is supported by matched timing/NCU evidence; Week 3's second tile-choice optimization is supported by stable three-session benchmarking and matched NCU captures; Week 4's reproducibility, limitations, contribution attribution and final report are complete. The separate literal 972-image transcription is still an owner-requested add-on, not a missing PDF objective. See [current-run evidence](CURRENT_RUN_2026-11-02.md) and the [final report](SUBMISSION_REPORT_2026-11-03.md).
