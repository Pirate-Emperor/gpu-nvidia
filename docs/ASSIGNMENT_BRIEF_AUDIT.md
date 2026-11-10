# Assignment brief audit — 5 November 2026

**Source of truth:** repository-root `nvidia2-cutile.pdf`  
**Source title:** *Course Project: Fused Attention with cuTile*  
**PDF:** 6 pages, 147,208 bytes  
**SHA-256:** `b4bdef5a0ede51c5661dfc006b8556a437c23764476e0fee656bb907fb030946`  
**Audited source revision:** `54d8e468ef991ae724f77fcf150149a1923db826` (the PDF itself is unchanged in the subsequent documentation-only revisions).

This audit is a direct page-by-page check of the original PDF, not an inference from the older project checklist. The PDF was extracted and all six rendered pages were visually checked. The source PDF and extracted text are preserved in GitHub Actions artifact [#13156007173](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41473591143/artifacts/13156007173), from workflow [#41473591143](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41473591143).

## Current-source verification update — 5 November 2026

The original checklist rows below were first drafted before the latest RTX 4060 validation. The detailed [current-run evidence record](CURRENT_RUN_2026-11-02.md) supersedes their pre-run “queued” and “pending” status text.

- Expanded current-source native workflow [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189) passed: genuine cuTile smoke, **57 native GPU correctness tests**, 3 optional GPU tests, CLI/NumPy check, 29/29 numerical case/tile configurations with zero tolerance violations, three benchmark sessions, clean-cache first-call measurements, framework reference, hint sweep and two standard NCU captures.
- Hosted CPU suite: 801 passed, 70 skipped. Skips are not counted as GPU validation.
- Independent D/Dv behavior and contract boundary checks are verified; host and direct-launch paths enforce PDF B/H/N limits.
- The 29-row current-source benchmark includes 13 rows meeting ≤5% fused session spread and ≤5% baseline-bracket drift. Noisy rows are retained but are not headline claims.
- Clean-cache first-call/launch timing is measured for long D=128 / 64×64 (2170.951 ms) and longer-batched D=128 / 64×128 (3059.894 ms), both correct versus NumPy.
- Week 2 first profiler-guided optimization remains the targeted `LOAD_LATENCY=10` hint for medium D=128 / 64×64; the expanded 5 November sweep reports a 12.68% gain against its compiler-default median with a 2.25% candidate session range. Longer-batched D=128 still keeps hint 0.
- Focused framework workflow [#41511770773](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41511770773) passed: CUDA FP32 PyTorch SDPA vs NumPy on seven workloads (7/7 pass, max absolute discrepancy 0.00048828125), and FP16 efficient SDPA on 10 configurations × 3 sessions (30/30 correctness checks). Backend: aten::_efficient_attention_forward. This is supplementary; speedup grading remains against unfused FP32 CuPy.
- Full corrected SASS/PTX pane movement proof is published: 27 profiles, 216 pane checks, zero failures and 486 PNGs in 27 profile archives. Literal manual transcription of all 972 original screenshots remains incomplete.
- Final 5–8 page report/PDF and repository-based contributor attribution are complete. Off-repository contribution percentages are not inferred.

## Requirements and current evidence

| Original brief requirement | Current repository evidence | Status |
|---|---|---|
| Genuine NVIDIA cuTile implementation of fused causal scaled dot-product attention | `src/cutile_attention.py`; the kernel tiles Q/K/V, masks future keys, maintains online-softmax state, accumulates P×V and writes output in one launch | Implemented and verified in the current RTX 4060 run #42075683189 |
| FP16 Q/K/V input, FP32 accumulation, FP16 or documented FP32 output | Kernel uses FP32 score/softmax/P×V accumulation and FP16 output; the updated path carries D and Dv independently, with unequal-dimension tests added | Implemented; unequal-D/Dv GPU oracle cases passed in run #42075683189 |
| `D` and `Dv` each supported in `{32,64,128}` | Equality checks were removed from `src/contract.py` and `src/cutile_attention.py`; CPU contract and native GPU oracle tests now cover unequal legal dimensions. The current CPU suite and native RTX 4060 run #42075683189 passed. | **Implemented and native-GPU verified** |
| Bounds `1≤B≤16`, `1≤H≤32`, `1≤N≤8192`; self-attention `Nq=Nk=N` | Assignment bounds are enforced in the shared host validator and direct device-launch path; CPU tests exercise legal boundaries and just-outside values. The current native GPU workflow passed. | **Implemented and native-GPU verified** |
| Reject unsupported or non-tile-divisible inputs clearly | Shared host validation and direct device-launch validation reject unsupported D/Dv, invalid B/H/N bounds and query/key tile non-divisibility; the native matrix exercises all 29 legal case/tile rows and contract tests cover just-outside bounds | **Implemented and tested** for the assignment contract; partial edge tiles are explicitly outside the fixed-tile scope |
| CLI command with NPZ input, NPY output and `--backend cutile` | `run_attention.py`, `src/attention_io.py`, deterministic input generation and independent output validation scripts | **Implemented and current-source verified**; CLI/NumPy oracle passed in native run #42075683189 |
| Correctness across D, multiple batches/heads/lengths and tiles; fixed seeds; small/large values; NaN/Inf | Independent NumPy FP32 reference, scalar/tiled references, fixed-seed cases, edge tests; historical native evidence reports 48 GPU tests and 29 legal case/tile configurations passed | Current-source native GPU validation passed in run #42075683189; see current-run record |
| Error reporting with maximum absolute/relative error and mixed tolerance | Reported tolerance `abs(error) <= 0.02 + 0.02*abs(reference)`, error summaries and zero violations in historical native run | Historical evidence exists; regenerate from current source |
| Unfused baseline and vendor/framework attention reference | Unfused FP32 CuPy baseline; CUDA FP32 PyTorch SDPA vs NumPy on seven workloads; and three-session FP16 efficient SDPA on ten configs. Run #41511770773 recorded aten::_efficient_attention_forward; all 30 paired checks passed. | **Complete; supplementary reference, not the speedup denominator** |
| At least six representative tile-divisible configurations, including short/long and D=64/D=128; warmup; ≥30 CUDA-event samples; median, speedup and effective output throughput | Seven workload families, 29 legal case/tile rows, three sessions, ten warmups, 30 timed samples and per-session output-throughput records exist | Current-source three-session run and 29-row table are complete. Thirteen rows meet the conservative spread/drift filter; unknown clock/power details are labelled unavailable. |
| Report first-call/JIT separately from steady-state; clean-cache timing when possible | First-call/JIT-associated fields were recorded where available, but long specializations were already compiled and show null values | Complete: two long specializations measured from empty cuTile/driver caches; first-call wall time is separated from steady state. |
| State GPU, CUDA/DSL versions, clocks/power policy if known and compilation mode | GPU, driver, Python, CuPy/cuTile, Nsight and compilation-mode details are recorded. The current run includes a sampled 2700 MHz SM clock and 115 W power limit; power draw/fixed-clock policy are explicitly unavailable. | **Complete with limitations documented**; no unavailable power draw or locked-clock policy is invented. |
| Nsight Compute on at least two configurations; discuss global traffic, occupancy, arithmetic and memory throughput, and bottleneck | 27 native RTX 4060 reports; Details/Raw exports; occupancy, scheduler, compute/DRAM and memory metrics; 27 per-configuration analyses; full-matrix four-pane movement proof: 216/216 checks | **Complete for PDF scope**; literal screenshot transcription is a separate owner-requested add-on |
| Week 2: first profiler-guided optimization | Current-source six-value LOAD_LATENCY sweep and matched 64×64 medium-D128 NCU pair; correctness checked before timing | **Complete:** hint 10 improved the three-session median by 12.68% (candidate range 2.25%) and reduced matched NCU duration from 52.80 to 47.49 µs. The higher shared-memory use and unchanged ~8.3% occupancy are documented; hint 10 remains workload-specific. |
| Week 3: second profiler-guided improvement and remaining bottleneck | Earlier records only established the Week 2 `LOAD_LATENCY=10` hint. A separate current-source tile-choice loop was added for longer-batched D=128 and compared 64×64 with 64×128. | **Complete:** 64×128 improved the stable three-session median by 1.60%; matched NCU duration was 0.68% lower and throughput slightly higher. The choice costs +28 registers/thread and +16.38 KiB static shared memory/CTA, with occupancy unchanged at 8.33% and slightly worse scheduler eligibility. It is accepted only for this workload; global default stays 64×64. See [Week 3 NCU note](ncu/rtx4060/WEEK3_TILE_CHOICE_2026-11-05.md). |
| Final report 5–8 pages/Markdown with derivation, decomposition, tensor/tile layout, correctness, benchmarks, plots, profiler, history, limitations and individual contributions | Final A4 report in [SUBMISSION_REPORT_2026-11-03.md](SUBMISSION_REPORT_2026-11-03.md) and [PDF](SUBMISSION_REPORT_2026-11-03.pdf); contribution statement bounded to verifiable Git author history | **Complete**; off-repository contributor allocations are not invented |
| README/setup, CLI, kernel, reference, unfused baseline, tests, benchmark scripts and raw CSV/JSON | Updated root README and source/tests/benchmarks/results; current-source native, NumPy/PyTorch FP32 and efficient-SDPA evidence | **Complete**; see CURRENT_RUN_2026-11-02.md and final report |

## Contract gaps identified and corrected in code

The original review found the following implementation gaps. They were corrected on `main` in the implementation change `e5e0bf8`; the CPU contract tests pass, while the current native RTX 4060 validation passed in workflow #42075683189. Read these as historical findings plus present verification status, not as outstanding code restrictions.

### 1. Independent value dimension

The PDF's tensor definition on page 1 uses `V[..., Dv]` and output `O[..., Dv]`. Page 3 lists `D,Dv ∈ {32,64,128}`; it does not state `Dv=D`. The earlier equality checks in `src/contract.py` and `src/cutile_attention.py` have been removed. The new CPU contract tests and GPU reference-comparison tests cover unequal legal pairs such as (32,64), (64,128) and (128,32). The native RTX 4060 GPU oracle matrix has now passed unequal-D/Dv cases.

### 2. B/H/N bounds

Page 3 sets `1≤B≤16`, `1≤H≤32`, `1≤N≤8192`. Explicit checks were added in the shared host validator and direct device-launch path. CPU tests cover legal minima/maxima and out-of-range values. Current native RTX 4060 validation completed successfully.

## Current NCU evidence and owner-requested screenshot review

The existing 27 native profiles are actual Nsight Compute captures from the RTX 4060 / SM89. The 972 existing GUI screenshots are 864 primary images (32 per report) plus 108 PTX supplement images (4 per report), each 1936×1048. The native profiles need not be re-run merely to fix GUI navigation.

However:
- The old SASS and PTX right-code/right-metrics horizontal screenshot pairs are SHA-256 identical across all 27 profiles. They cannot be described as proof that those right panes moved.
- The earlier 32+4 image inventory and metric write-ups are not a literal manual transcription of every visible instruction/metric in every screenshot.
- One corrected horizontal probe passed on `long_d128-tq128-tk128-lat0` in [run #41478616763](https://github.com/xemptynothing/gpu-nvidia/actions/runs/41478616763): all four crop regions changed in both SASS and PTX modes and returned exactly to baseline on inverse scroll. That proves the mechanism on one report only. Follow-up #41479365336 has now been validated and published in commit 058e6ea: 216 pane checks, zero failures, 486 images. Do not confuse this with the still-incomplete literal manual transcription of all 972 original views.

See [the 27-profile GUI analysis index](ncu/rtx4060/27-profile-gui-analysis/README.md) and [capture audit](ncu/rtx4060/27-profile-gui-analysis/CAPTURE_AUDIT.md).

## Follow-up on contract validation

The two contract issues below were identified by the page-by-page audit and have now received code changes on `main` (implementation revision `e5e0bf8`). The 2 November CPU suite passes 801 tests with 70 skips; the skips include GPU modules unavailable on the hosted runner without CuPy/Torch and reference cases filtered because their tiles do not divide the sequence. The current native RTX 4060 validation passed; these corrections are now verified implementation requirements.

### 1. Independent value dimension

The PDF's page 1 tensor definition uses `V[..., Dv]` and output `O[..., Dv]`. Page 3 lists `D,Dv ∈ {32,64,128}`; it does not state `Dv=D`. The explicit equality checks were removed from `src/contract.py` and `src/cutile_attention.py`. The kernel already carried `head_dim` and `value_dim` separately through its K/V tile and output shapes; CPU contract tests and native GPU cases now cover unequal legal pairs such as (32,64), (64,128) and (128,32). The native GPU test job passed.

### 2. B/H/N bounds

Page 3 sets `1≤B≤16`, `1≤H≤32`, `1≤N≤8192`. The host launch validator and direct device-launch path now enforce those bounds. CPU tests cover legal minima/maxima and out-of-range values. Current native GPU confirmation passed in workflow #42075683189.

## Final assignment disposition — 5 November 2026

**The six-page course assignment objectives are complete.** The kernel and assignment tensor/shape contract have been tested on the RTX 4060; the FP32 NumPy oracle and CUDA FP32 PyTorch SDPA reference agree within the published mixed tolerance; the expanded 57-test native GPU suite, 29-row numerical matrix, CLI, three-session benchmark, clean-cache first-call experiment, framework comparison, 27-profile NCU analysis, two distinct matched profiler-guided optimization loops, reproducibility evidence and final 5–8 page report are present.

The final report is [SUBMISSION_REPORT_2026-11-03.md](SUBMISSION_REPORT_2026-11-03.md) and [SUBMISSION_REPORT_2026-11-03.pdf](SUBMISSION_REPORT_2026-11-03.pdf). The second Week 3 tile-choice evidence is [NCU run #42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947), artifact [#13440087603](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947/artifacts/13440087603). The only explicitly tracked item left open is the additional owner-requested literal transcription of every detail in the 972 original screenshots. It is not required by the assignment PDF; the required profiler analysis and full-matrix pane movement proof are complete.
