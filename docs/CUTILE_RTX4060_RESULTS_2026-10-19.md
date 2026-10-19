# Actual cuTile RTX 4060 results — 2026-10-19

**Outcome:** fused causal-attention correctness, controlled benchmarking, and two-configuration Nsight Compute profiling PASS on the native Windows RTX 4060. Both basic resource/throughput reports and supplemental scheduler/warp-state reports were captured and inspected; see the [profile analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) and [artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).

## Canonical workflow and preserved evidence

- Workflow: [RTX 4060 native Windows cuTile validation #39160827910](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910)
- Tested commit: 12120f8a67a1fbd6f9cd83bd8f2f8bf7b31ce428. The environment log records matching WORKFLOW_SHA and CHECKOUT_GIT_HEAD.
- Raw artifact: [cutile-rtx4060-experiment.zip, artifact 12148414465](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910/artifacts/12148414465)
- Artifact SHA-256: c84b781635a23571139ca8734df9e9d39ddab8485bfb5875c30db2ec63361854
- The ZIP contains environment and pip-freeze records, smoke and test logs, numerical error JSON/log, CLI input and output with oracle-verification log, three raw benchmark JSON/CSV/log sessions, the aggregate benchmark JSON/CSV/log, and a profiling status log. It contains no NCU profile reports.

## Environment

| Item | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, compute capability 8.9 |
| Driver | 616.92 |
| VRAM | 8,188 MiB reported total |
| OS | Windows x64, build 26300; Python platform string Windows-11-10.0.26300-SP0 |
| Python | 3.12.10 |
| NumPy | 2.5.4 |
| CuPy | 14.2.0, cupy-cuda13x with ctk extra |
| NVIDIA cuTile Python | cuda-tile 1.6.0 |
| Tile compiler packages | TileIRAS, NVCC and NVVM 13.4.92 |
| CUDA driver API reported by CuPy | 13040 |
| CUDA runtime API reported by CuPy | 13020 |
| pytest | 8.4.2 |
| Project virtual environment | C:\Users\Public\GPU-cutile-project\gpu-nvidia\.venv-cutile-windows |

All Python dependencies were installed into the project-local virtual environment. Pip cache, temp/extraction files, CuPy cache and CUDA cache were contained under project-root .cache and .tmp. The original correctness/benchmark run did not install a profiler or system-wide CUDA Toolkit. After explicit user approval, Nsight Compute 2026.3.1.0 was installed separately and then used by the focused profiling workflow; those reports are archived separately in artifact 12149902039. The profiling workflow downloaded no additional tools or datasets and used no administrator privileges.

## Correctness and command-line gates

The actual cuTile suite ran with REQUIRE_CUTILE=1: **48 passed in 7.05 seconds, zero failed**. Missing CuPy/cuTile imports are hard errors under this flag rather than silent skips.

The separate numerical diagnostics checked **29 legal case/tile combinations**, all against the independent FP32 NumPy full-matrix oracle, with zero violations of the fixed mixed tolerance:

    abs(actual - reference) <= 0.02 + 0.02 * abs(reference)

| Accuracy metric across the 29 configurations | Result |
|---|---:|
| Maximum absolute error | 0.0009765625 |
| Largest per-configuration P99 absolute error | 0.00048828125 |
| Largest per-configuration RMSE | 0.000086860741 |
| Maximum elementwise error / permitted tolerance | 0.0243665 |
| Largest relative error where absolute reference is at least 0.01 | 0.0224587 |
| Total tolerance violations | 0 |

The max tolerance ratio means the worst element used about 2.44% of the permitted error envelope; it is not a claim of 2.44% absolute-output error. Every per-case result, element count and tolerance count is preserved in results/cutile-accuracy.json.

The CLI generated deterministic FP16 Q/K/V at shape [1, 1, 64, 32], seed 20261018, and used tile 64x64. The actual run_attention.py cuTile backend wrote an FP16 output file; scripts/verify_cli_output.py checked shape, dtype, finiteness and agreement with the independent NumPy oracle, and printed CLI_REFERENCE_CHECK=PASS.

## Performance methodology

Three independent sessions used the seven frozen workloads and five legal tiles. Tile orders were rotated to avoid always favoring the first configuration:

1. 32x64, 64x64, 64x128, 128x64, 128x128
2. 128x128, 128x64, 64x128, 64x64, 32x64
3. 64x64, 32x64, 128x64, 64x128, 128x128

Each configuration/session used ten warmups and 30 CUDA-event samples. Each event sample covered five launches, divided by five, yielding 150 timed launches per configuration per session. Unfused baseline medians/P90s were measured before and after each case's tile sweep; their average is reported as the centered baseline, and the pre/post drift is retained. First-call wall time is kept separate from steady-state events; it may include compiler cache hits and should be interpreted as first-call/JIT-associated latency, not pure compiler-only time.

The baseline is the repository's intentionally unfused FP32 CuPy attention implementation on resident GPU inputs, which materializes full score and probability intermediates. The fused cuTile measurement includes output allocation and the device launch but excludes host-to-device copies. The comparison is against this unfused baseline, not against an optimized vendor attention kernel. A CUDA-enabled PyTorch SDPA comparison was not run because a CUDA PyTorch build is not installed in the project virtual environment.

Times below are milliseconds. Session spread is (maximum median - minimum median) divided by median across three sessions. Baseline drift is the largest pre/post baseline median difference divided by their average. Output throughput fields are preserved in each raw benchmark JSON; these use output bytes only and must not be described as total memory bandwidth.

| Case | Tile | Fused median | Fused P90 | Unfused baseline | Speedup | 3-session spread | Max baseline drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| long_d128 | 32×64 | 0.335770 | 0.346829 | 4.414310 | 13.15× | 2.6% | 0.5% |
| long_d128 | 64×64 | 0.267059 | 0.271626 | 4.414310 | 16.53× | 2.0% | 0.5% |
| long_d128 | 64×128 | 0.272794 | 0.277709 | 4.414310 | 16.18× | 2.6% | 0.5% |
| long_d128 | 128×64 | 0.318771 | 0.321454 | 4.414310 | 13.84× | 2.2% | 0.5% |
| long_d128 | 128×128 | 0.412160 | 0.419860 | 4.414310 | 10.71× | 1.7% | 0.5% |
| longer_batched_d128 | 32×64 | 1.158758 | 1.177887 | 17.131469 | 14.77× | 1.3% | 0.1% |
| longer_batched_d128 | 64×64 | 0.939622 | 0.948736 | 17.131469 | 18.21× | 1.5% | 0.1% |
| longer_batched_d128 | 64×128 | 0.923034 | 0.937656 | 17.131469 | 18.54× | 0.9% | 0.1% |
| longer_batched_d128 | 128×64 | 1.022259 | 1.034281 | 17.131469 | 16.76× | 0.2% | 0.1% |
| longer_batched_d128 | 128×128 | 1.295872 | 1.305272 | 17.131469 | 13.21× | 0.8% | 0.1% |
| medium_d128 | 32×64 | 0.066150 | 0.066601 | 0.381338 | 5.77× | 9.1% | 4.0% |
| medium_d128 | 64×64 | 0.055194 | 0.057979 | 0.381338 | 6.89× | 10.8% | 4.0% |
| medium_d128 | 64×128 | 0.057139 | 0.061030 | 0.381338 | 6.67× | 0.7% | 4.0% |
| medium_d128 | 128×64 | 0.077722 | 0.079258 | 0.381338 | 4.92× | 7.2% | 4.0% |
| medium_d128 | 128×128 | 0.105574 | 0.108790 | 0.381338 | 3.59× | 1.2% | 4.0% |
| medium_d64 | 32×64 | 0.019331 | 0.021791 | 0.290458 | 15.03× | 4.8% | 1.7% |
| medium_d64 | 64×64 | 0.019763 | 0.022753 | 0.290458 | 15.20× | 19.7% | 1.7% |
| medium_d64 | 64×128 | 0.018432 | 0.019313 | 0.290458 | 15.76× | 5.0% | 1.7% |
| medium_d64 | 128×64 | 0.022835 | 0.023757 | 0.290458 | 12.83× | 1.3% | 1.7% |
| medium_d64 | 128×128 | 0.022630 | 0.026214 | 0.290458 | 12.70× | 6.4% | 1.7% |
| short_d32 | 32×64 | 0.016806 | 0.019290 | 0.319893 | 18.84× | 184.6% | 87.2% |
| short_d32 | 64×64 | 0.015565 | 0.018657 | 0.319893 | 20.55× | 180.1% | 87.2% |
| short_d64 | 32×64 | 0.025907 | 0.032645 | 0.324186 | 12.75× | 99.6% | 3.9% |
| short_d64 | 64×64 | 0.023552 | 0.026092 | 0.324186 | 14.03× | 142.6% | 3.9% |
| short_multihead_d64 | 32×64 | 0.016179 | 0.020357 | 0.295885 | 17.73× | 9.6% | 5.3% |
| short_multihead_d64 | 64×64 | 0.016170 | 0.019538 | 0.295885 | 18.10× | 4.5% | 5.3% |
| short_multihead_d64 | 64×128 | 0.016179 | 0.018470 | 0.295885 | 18.44× | 3.1% | 5.3% |
| short_multihead_d64 | 128×64 | 0.017306 | 0.020685 | 0.295885 | 17.10× | 18.3% | 5.3% |
| short_multihead_d64 | 128×128 | 0.018637 | 0.021996 | 0.295885 | 16.10× | 10.4% | 5.3% |

## Optimization decisions

The current kernel contains two changes that passed the fixed correctness gate:

1. **Causal loop bound:** each query block reads only key/value tiles that can contain a key at or before its final query position. Entirely future tiles are skipped instead of being loaded and multiplied only to be masked.
2. **P×V operand type:** QKᵀ accumulation, online-softmax state, row normalizers and P×V accumulator remain FP32; probabilities are rounded to FP16 for the second MMA and the final output is FP16. The 48-test suite and all 29 numeric reports passed this choice under the frozen tolerance.

Use **64×64 as the general default**. It was best on long D=128 (0.267059 ms, 16.53× versus the unfused baseline). For the longer batched D=128 case, **64×128** was fastest at 0.923034 ms (18.54×), about 1.8% faster than 64×64 with 0.9% spread and 0.1% baseline drift. Treat the specialization as provisional until another independent session confirms its advantage. The 128×128 configuration loses on the long D=128 cases and should not be promoted.

Do not headline the short_d32/short_d64 speedups: Windows/WDDM timings varied massively between sessions, and short_d32 baseline drift reached 87.2%. Medium D=128 fused spread reached 10.8%, so tile ranking there is also provisional.

## Nsight Compute profiling — PASS (2026-10-19)

Two genuine cuTile reports were captured on the Windows RTX 4060 with Nsight Compute CLI 2026.3.1.0. The GPU accepted hardware-counter collection; there was no RTX 4060 `ERR_NVGPUCTRPERM`.

- Full native run, including the repeated smoke/correctness/CLI/benchmark path: [workflow #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223).
- Final reports and metric exports: [profile artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039), produced by [profile run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066).
- Detailed interpretation: [CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md).

| Profile | Kernel duration | Registers/thread | Static shared memory/block | Achieved occupancy | Compute throughput | DRAM throughput |
|---|---:|---:|---:|---:|---:|---:|
| medium_d128, 64×64 | 52.90 µs | 231 | 65.54 KiB | 8.35% | 18.81% | 22.88% |
| longer_batched_d128, 64×128 | 984.26 µs | 255 | 81.92 KiB | 8.33% | 30.27% | 11.89% |

The basic Occupancy section identifies shared memory as the CTA-residency limiter (one CTA per SM, about 8.3% achieved occupancy). The supplemental Scheduler/ Warp State sections recorded no-eligible-warp cycles of 85.83% and 85.26%, respectively, and Nsight's execution-pipeline-stall rule estimated 2.4 / 3.1 cycles per issued instruction. The rule's estimated local speedups (33.57% / 46.04%) are heuristic outputs, not measured gains. The optional `smsp__pcsamp_sample_count` metric was unavailable, so detailed source-level stall sampling could not be inspected.

**Optimization decision:** retain the validated 64×64 general default and 64×128 longer-batched specialization. The reports make shared-memory/register pressure and instruction scheduling the evidence-based next research targets, but do not justify an untested change by themselves. Any new kernel variant must pass correctness first and then repeat the controlled benchmark sessions before promotion.

The raw compressed `.ncu-repz` reports, exported details and raw CSVs are preserved in artifact 12149902039. Historical A100 `ERR_NVGPUCTRPERM` evidence is separate and does not describe the RTX 4060 result.
