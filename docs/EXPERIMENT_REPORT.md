# Fused Attention with cuTile — Experiment Report

**Updated:** 2026-10-19 (local runner audit; historical benchmark numbers unchanged)  
**Repository:** [SpryzenHell/gpu-nvidia](https://github.com/SpryzenHell/gpu-nvidia)  
**Core kernel snapshot covered:** `4485bdce5ed462dfbf495b8129efce91b4124585`; the paired v5 benchmark uses code at `caeb6e79e4237222d021782a3a9bdb72d6c4ca7f` (documentation updates are committed afterward)


## Current verified result — 2026-10-19 actual cuTile attention on RTX 4060

The previous runner-discovery table below documents failures before the project venv existed. It is historical and superseded by [successful workflow #39160827910](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910), commit 12120f8a67a1fbd6f9cd83bd8f2f8bf7b31ce428.

| Gate | Current result | Evidence |
|---|---|---|
| Genuine cuTile smoke | PASS | Vector-add compile/launch, synchronization and exact output validation passed. |
| Attention correctness | PASS | 48 passed / 0 failed with REQUIRE_CUTILE=1. |
| Numerical diagnostics | PASS | 29 legal case/tile configurations, zero tolerance violations. |
| CLI / NumPy oracle | PASS | FP16 output contract and independent reference check passed. |
| Performance comparison | PASS with caveats | Three repeat sessions, five tile configurations, 10 warmups and 30 CUDA-event samples with five launches/sample; raw JSON/CSV archived. |
| Actual-cuTile NCU profiles | OPEN / BLOCKED | No ncu executable under project root or .tools; no download or profile attempted. |

All evidence is in [artifact 12148414465](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910/artifacts/12148414465) and the [detailed current results](CUTILE_RTX4060_RESULTS_2026-10-19.md).

The current fused kernel skips wholly-future K/V tiles and uses FP16 probability operands for P×V with FP32 accumulation and online-softmax state. Use 64×64 as the general default; 64×128 measured best for the longer batched D=128 case. The 128×128 tile loses on long D=128 cases. Short sequence timings are extremely noisy on Windows/WDDM and should not be used as headline speedup claims.

The remaining hard gate is two actual cuTile Nsight Compute report files, medium D=128 at 64×64 and longer batched D=128 at 64×128. Tool download needs owner approval for project-local .tools. The CUDA-enabled PyTorch SDPA comparison remains unrun; this benchmark uses the deliberately unfused FP32 CuPy baseline.


## Historical runner setup failures — before local venv creation (superseded by the successful run above)

These first attempts happened before the project-local venv existed. They are retained as chronological history; later run #39160827910 executed and passed actual attention correctness and the controlled benchmark.

| Run ID | Commit (short) | Outcome | Exact finding |
|---|---|---|---|
| [39154733638](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154733638) | 87381e42 | Failed before GPU preflight | Windows PowerShell Core command pwsh absent; all project experiments skipped. |
| [39154791220](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154791220) | 1cfc16dd | GPU identified; environment path missing | nvidia-smi: NVIDIA GeForce RTX 4060, driver 616.92, CC 8.9, 8188 MiB. Expected project-local .venv-cutile-windows interpreter absent. No install attempted. |
| [39154888145](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154888145) | 2971f5ac | No in-repository Python environment | Python Launcher found the global CPython 3.12 interpreter but its package probe produced an import traceback. |
| [39154979187](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979187) | b9189b5d | Confirmed blocker | Probe error captured explicitly: ModuleNotFoundError: No module named 'numpy'. Artifact [12146591326](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979187/artifacts/12146591326) preserves the environment discovery log. |

At that historical point, the workflow only searched for an existing environment and no install was attempted. After the user created the project-local venv and allowed packages to be installed there, the workflow was updated to install dependencies inside that venv, and run #39160827910 completed the attention, CLI, accuracy, and benchmark gates.

**Remaining work after the successful run:** capture and inspect two actual-cuTile Nsight Compute reports after an official profiler is approved for project-local .tools. Correctness, CLI, numerical diagnostics, and performance metrics are now available above and in the linked artifact.

## Historical executive status before local cuTile validation

| Area | Result | Meaning |
|---|---|---|
| Independent math references | PASS | FP64 scalar oracle, FP32 full-matrix oracle, and FP32 online recurrence provide separate checks. |
| Native CUDA fused path on A100 | PASS | Base suite passed on A100/R550; both visible A100 devices are covered by device-matrix tests. |
| Broad non-cuTile regression | PASS | [Run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082): **907 passed, 52 skipped, 0 failed**. |
| CPU/reference suite | PASS | [Run #334](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264023): **763 passed, 68 skipped, 0 failed**. GPU-only tests skip on hosted CPU CI when dependencies/devices are absent. |
| Experimental v4 | PASS correctness; NOT promoted | [Run #1](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760982099): **30 passed**; performance mixed; PTXAS output equivalent to current. |
| Experimental v5 | PASS correctness; repeated benchmark complete; NOT promoted | [Run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163): **30 passed**; three order-alternating paired sessions per case/tile. Results are mixed/noisy on short cases; long/batched gains were only about 0.3–0.6%. |
| A100 timing harness | PASS | CUDA events, warmups, 30 measured iterations, medians/P90; cold compile timed separately in the version ladder. |
| Nsight Compute profiling | BLOCKED | [Run #1](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760475268) fails with `ERR_NVGPUCTRPERM`; no kernel was profiled. |
| Nsight Systems timeline (ordinary-CUDA surrogate) | PASS | [Run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950) captured two warmed traces and exported CUDA kernel/API summaries; [artifact](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954). These are not actual-cuTile profiles. |
| Actual NVIDIA cuTile on A100 | BLOCKED | Driver is R550.90.07; current project gate requires R580+. The CUDA surrogate is not actual cuTile. |

**Important distinction:** ordinary-CUDA correctness and performance results are useful engineering evidence, but do not satisfy the final actual-cuTile execution/profiling requirement.

## Historical continuation review before the native Windows run

- At the start of this continuation, `main` was verified at `e87d8b9352fdf6783347d5c1e1945172011ebeb9`, matching the handoff commit.
- The existing A100 runner's verified driver is R550.90.07, below NVIDIA's documented R580+ cuTile Python runtime minimum. The actual-cuTile route moves to the user's local GPU after inventory, or another explicitly authorized supported environment. Retain the A100 for ordinary-CUDA surrogate controls only; do not assume local hardware or repeat the failed compatibility path.
- NVIDIA's [current cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) lists Linux x86_64/Linux aarch64/Windows x86_64, compute capability 8.x–12.x, driver R580+, and Python 3.10–3.14. The `[tileiras]` path provides compiler components in the venv with matching major.minor package versions; system CUDA Toolkit 13.1+ is an alternative. Detailed profiling requires Linux R580.126.09+ or Windows R582.16+.
- **Historical Actions access note (2026-10-18):** the connected API did not expose several handoff IDs at that time. That earlier API limitation was resolved; the current run #39160827910 and its logs/artifact have been verified.
- At that earlier point, the local GPU inventory had not yet been supplied, so no Windows-native/WSL2 decision, runner registration, isolation claim or actual GPU test had been made.

## 2. Historical A100/R550 hardware and software constraints

Verified self-hosted runner:
- Two visible NVIDIA A100-SXM4-40GB GPUs; SM80 / compute capability 8.0; MIG disabled.
- Driver R550.90.07, CUDA Toolkit/NVCC 12.4.131, GCC 11.4.
- CuPy RawKernels use NVCC with `-arch=sm_80`.
- Runner is non-root and has no passwordless sudo. No driver upgrade is attempted by CI.

### Environment approaches tried

| Approach | Observation | Decision |
|---|---|---|
| Use actual cuTile directly on R550 | Does not meet current R580+ driver gate. | Keep explicitly blocked; never label surrogate as cuTile. |
| `cuda-compat-13-4` | Package downloaded/extracted, but runtime first returned `cudaErrorInsufficientDriver`; loading compatibility libraries returned CUDA error 803, system driver mismatch. | Rejected. |
| `cuda-compat-13-0` + CUDA 13.0 runtime/CuPy | Package artifacts were obtained and tested as an alternative, but no verified successful end-to-end cuTile GPU run was established. | Do not count as a working workaround. |
| Early `cuda-tile 1.2.0` attempt | Package was not available from the configured package index; experiment did not reach GPU execution. | No correctness/performance result. |
| Fix native-CUDA JIT path | Initial RawKernel compilation had a header-discovery problem under NVRTC. Native setup was adjusted to use NVCC and matching CUDA runtime headers; basic CUDA and later fused-kernel workflows passed. | Retain NVCC-based native path. |
| Nsight Compute | Executable/version detected, but profiler returned `ERR_NVGPUCTRPERM` and “No kernels were profiled.” | Profiling access remains blocked; do not claim profiler success. |

## 3. Correctness methodology

The checks are deliberately independent to reduce self-validation:

1. **Explicit scalar FP64 oracle** — computes the causal attention equation with explicit loops.
2. **Full-matrix FP32 NumPy oracle** — FP16 input conversion, QKᵀ, (1/\sqrt{D}), causal mask, max-shift softmax, and (PV).
3. **Tiled online-softmax FP32 reference** — independently checks the running-max/running-norm/output recurrence, including all-future tiles.
4. **GPU implementation tests** — compare kernel output to references and to independent metamorphic properties.

Coverage includes:
- D=32/64/128, multiple N/B/H and supported tile sizes 16/32/64 for the shared-tile CUDA path.
- Explicit FP64 recurrence cross-checks for small N.
- Causal future-K/V invariance; current-query locality; constant V; zero-logit prefix means; extreme finite logits; batch/head isolation; affine V transform; common key shift; tile-to-tile and cross-device agreement.
- Validation of FP16 dtype, rank/shape/contiguity, finite inputs, required dimensions, tile divisibility and invalid tile sizes.
- Long sequence cases through N=2048, D=128.

The FP16-output comparisons use mixed absolute/relative tolerance where numerical rounding matters. Exact equality is reserved for deterministic invariants where it is mathematically appropriate.

### Test history and failure classification

- Earlier native runs #38/#39 completed at **128 passed, 1 skipped**.
- Expanded run #43 reported **132 passed, 1 failed, 1 skipped**. The failure was a test bug: affine-V used `tile_k=2`, which the kernel correctly rejects.
- Fixed the test to use a legal `tile_k=16` case. Native run #44 then passed **133 tests, 1 skipped**; full run #47 passed **905, 52 skipped**.
- Current broad non-cuTile run #51 again passed **905, 52 skipped**.
- The full-suite skips are intentional/non-applicable cases (including non-dividing tile combinations) plus the optional PyTorch test when PyTorch is absent. CPU CI has additional GPU-only skips due to absent CuPy/GPU.
- Isolated v4 and v5 GPU suites each passed **30/30**. The broad suite's explicit test list does not include the v4/v5 test modules; those variants are validated by their dedicated A100 workflows.

## 4. Implementation approaches

| Approach | Key design | Status / finding |
|---|---|---|
| Independent references | FP64 scalar, FP32 matrix, FP32 tiled recurrence | PASS; the validation foundation. |
| Unfused CuPy baseline | Materializes score/probability tensors | Baseline for performance; not a fused target. |
| Naive scalar CUDA | Simple per-output CUDA path; two causal passes | Intentionally slow diagnostic baseline. |
| Block-row fused CUDA | One block/query row; cooperative dot-product reduction and stable two-pass softmax | PASS in native CUDA suite; no cuTile dependency. |
| Warp-row fused CUDA | One warp/query row; shuffle reductions | PASS in native CUDA suite; no cuTile dependency. |
| Shared-tile v1 | One query row/block, shared K/V tile, two-pass max/normalization | Correct; slow reference implementation within the tiled-kernel ladder. |
| Shared-tile v2 | Four query rows/block (warp per row), K/V in shared memory, online softmax | Correct and materially faster than v1 in the ladder. |
| Shared-tile v3 | v2 layout, computes four QK scores per inner iteration | Correct; faster on many short/medium cases but inconsistent for long D=128/batched cases. |
| Current shared-tile | v3-style four-score loop; reuses the Q fragment across those four score calculations | **Current A100 surrogate**; selected for reliable correctness and generally competitive measured latency. |
| v4: persistent Q registers | Attempts to keep Q fragments in registers across the whole K-tile traversal | Correct, but rejected as a promotion: PTXAS emits the same resource summary/cubin size as current, with mixed performance including a severe long-batched regression. |
| v5: `half2` QK loads | Processes adjacent FP16 Q/K values as `half2`, converting pairs to FP32 for accumulation | Correct, but not promoted yet; initial timings are mixed and too variable to prove a general win. |
| Actual cuTile kernel | Required course target; distinct backend/DSL | Implementation exists in `src/cutile_attention.py`, but not executed on this R550 host. Requires supported driver/runtime before it can be claimed validated. |

All shared-tile variants remain ordinary CUDA C++ through CuPy RawKernel/NVCC. They are **cuTile-like structural surrogates**, not NVIDIA cuTile.

## 5. Controlled timing evidence

The tile-ladder workflow uses 10 warmups, 30 CUDA-event timed iterations, median/P90, resident input tensors, and a fresh CuPy cache to measure first compile separately. [Ladder run #3](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760370849) reports first-compile measurements of:
- v1: **1652 ms**
- v2: **717 ms**
- v3: **1063 ms**
- current: **808 ms**

These are cold JIT/compile times for the tested environment, not steady-state kernel latency.

### Best observed median per approach within ladder run #3

Milliseconds; each entry is the lowest measured median among `tile_k=16/32/64` in that same run. v1 was intentionally run only on the first four smaller cases.

| Case | v1 | v2 | v3 | Current |
|---|---:|---:|---:|---:|
| short_d32 | 0.111 | 0.066 | 0.055 | 0.055 |
| short_d64 | 0.119 | 0.073 | 0.060 | 0.061 |
| short_multihead_d64 | 0.385 | 0.102 | 0.077 | 0.081 |
| medium_d64 | 1.727 | 0.234 | 0.180 | 0.193 |
| medium_d128 | — | 0.947 | 0.815 | 0.792 |
| long_d128 | — | 7.188 | 6.556 | 6.131 |
| longer_batched_d128 | — | 22.895 | 26.533 | 25.678 |

The follow-up v1/v2 workflow [run #23](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761551401) also shows a large structural gain from one-row/two-pass v1 to warp-row/online v2, but the absolute long-case medians differed noticeably across workflow runs. Treat these as **engineering measurements, not final polished performance claims**. Retest under controlled idle conditions and profile before publishing headline speedups.

### v4 and v5 outcomes

- **v4:** 30 correctness tests passed. It was near current for short cases, faster for some D=128 cases, and substantially slower on longer batched input for `tile_k=16/32`. PTXAS report [run #3](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761390319): v2/current/v4 use **32 registers, 33,792 bytes shared memory, zero reported spills**; current and v4 cubins are both 15,136 bytes. No resource-level improvement is evident.
- **v5:** dedicated [run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163) passed **30/30 correctness tests** and completed a paired benchmark across all seven cases and tile_k=16/32/64. Every version/session used 10 warmups and 30 CUDA-event samples; there were three sessions per case/tile, with the order alternating. The table reports median session medians. The ratio is current latency / v5 latency, so values above 1 favor v5; the range is the observed per-session ratio range.

| Case | tile_k | Current ms | v5 ms | Paired current/v5 | Observed range |
|---|---:|---:|---:|---:|---:|
| short_d32 | 16 | 0.050 | 0.051 | 0.980x | 0.960–0.980x |
| short_d32 | 32 | 0.064 | 0.052 | 1.276x | 0.971–1.402x |
| short_d32 | 64 | 0.049 | 0.050 | 0.980x | 0.980–1.353x |
| short_d64 | 16 | 0.075 | 0.058 | 1.304x | 1.000–1.465x |
| short_d64 | 32 | 0.057 | 0.057 | 0.982x | 0.794–1.018x |
| short_d64 | 64 | 0.057 | 0.061 | 0.933x | 0.775–1.018x |
| short_multihead_d64 | 16 | 0.085 | 0.085 | 0.976x | 0.950–1.012x |
| short_multihead_d64 | 32 | 0.084 | 0.093 | 0.906x | 0.764–1.232x |
| short_multihead_d64 | 64 | 0.082 | 0.084 | 0.988x | 0.947–1.000x |
| medium_d64 | 16 | 0.279 | 0.251 | 1.110x | 1.020–1.110x |
| medium_d64 | 32 | 0.276 | 0.254 | 1.116x | 0.909–1.123x |
| medium_d64 | 64 | 0.280 | 0.247 | 1.023x | 1.015–1.135x |
| medium_d128 | 16 | 0.933 | 0.908 | 1.024x | 1.018–1.028x |
| medium_d128 | 32 | 0.872 | 0.829 | 0.993x | 0.989–1.083x |
| medium_d128 | 64 | 0.807 | 0.806 | 1.002x | 0.993–1.002x |
| long_d128 | 16 | 5.257 | 5.239 | 1.003x | 1.003–1.004x |
| long_d128 | 32 | 5.192 | 5.170 | 1.004x | 1.004–1.005x |
| long_d128 | 64 | 5.152 | 5.129 | 1.004x | 1.003–1.005x |
| longer_batched_d128 | 16 | 20.421 | 20.368 | 1.004x | 0.996–1.004x |
| longer_batched_d128 | 32 | 20.172 | 20.065 | 1.004x | 1.004–1.008x |
| longer_batched_d128 | 64 | 20.017 | 19.902 | 1.006x | 1.003–1.007x |

Interpretation: v5 is **not promoted**. It improves medium D=64 in this run, but short inputs have mixed ratios, and long/batched gains are only about 0.3–0.6%. These results do not establish a robust general speedup; more independent sessions would be needed before promoting it.

The exact raw JSON/CSV benchmark artifacts are attached to their GitHub Actions runs. The repository keeps the benchmark harnesses and should preserve future dated artifacts under `results/runs/`.

## 6. Compiler resources and profiling

The updated PTXAS report [run #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) compiled v1/v2/v3/current/v4/v5 for SM80 and reported:

| Variant | Registers | Shared memory | Spill stores/loads | Cubin bytes |
|---|---:|---:|---:|---:|
| v1 | 28 | 33,544 B | 0 / 0 | 11,040 |
| v2 | 32 | 33,792 B | 0 / 0 | 15,392 |
| v3 | 40 | 33,792 B | 0 / 0 | 22,560 |
| current | 32 | 33,792 B | 0 / 0 | 15,136 |
| v4 | 32 | 33,792 B | 0 / 0 | 15,136 |
| v5 | 32 | 33,792 B | 0 / 0 | 15,520 |

A separate [Nsight Systems workflow #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950) completed and uploaded two `.nsys-rep` traces plus CUDA API/kernel summary outputs in [artifact #11671075954](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954): `medium_d128, tile_k=64` and `longer_batched_d128, tile_k=32`. These timelines are supplementary evidence for the ordinary-CUDA surrogate, not actual-cuTile profiles and not Nsight Compute hardware-counter metrics.

The exported `cuda_gpu_kern_sum` summaries report these **instrumented trace statistics**:

| Surrogate case | Tile | Kernel instances | Median duration | Mean duration | Min–max |
|---|---:|---:|---:|---:|---:|
| `medium_d128` | 64 | 7 | 0.977 ms | 0.998 ms | 0.975–1.126 ms |
| `longer_batched_d128` | 32 | 7 | 25.851 ms | 25.925 ms | 25.845–26.369 ms |

Each trace contains the initial call (which triggers JIT/compile), five warmups, and the final launch used as the profiling target. These values are **not steady-state benchmark results** and must not replace CUDA-event benchmark medians; profiling overhead and first-launch work are present. The CUDA API summary also shows substantial `cudaMalloc` activity during these short one-process traces, so API totals should not be interpreted as kernel latency.\n\nThis is compile-time resource evidence, **not a substitute for hardware profiling**. Nsight Compute could not collect counters because the runner account lacks GPU performance-counter permission; no occupancy, bandwidth, or instruction-counter conclusions can be drawn yet. v5 matches current on register count, shared memory, and zero reported spills, but its cubin is 384 bytes larger (15,520 B vs 15,136 B). This compiler-resource difference does not establish a runtime improvement.

## 7. Artifacts and code map

- References: `src/reference.py`, `src/online_reference.py`; scalar FP64 oracle in `tests/test_reference_exhaustive.py`.
- Core CUDA: `src/naive_cuda_attention.py`, `src/custom_cuda_attention.py`, `src/warp_cuda_attention.py`, `src/shared_tile_cuda_attention.py`.
- Historical variants: `src/variants/shared_tile_v1.py` through `shared_tile_v5.py`.
- Correctness: `tests/test_gpu_shared_tile_exhaustive.py`, `tests/test_gpu_kernel_versions.py`, `tests/test_gpu_kernel_v4.py`, `tests/test_gpu_kernel_v5.py`, plus reference/metamorphic suites.
- Benchmark drivers: `benchmarks/benchmark_kernel_versions.py`, `benchmark_kernel_v3.py`, `benchmark_tile_ladder.py`, `benchmark_v4.py`, `benchmark_v5.py`.
- Compiler resources: `scripts/ptxas_tile_report.py`.
- Native CUDA profiling driver/workflow: `benchmarks/profile_shared_tile.py`, `.github/workflows/a100-native-nsight.yml`.
- Existing target cuTile path/profiler: `src/cutile_attention.py`, `benchmarks/benchmark_attention.py`, `benchmarks/profile_fused.py`.

## 8. Historical decisions before actual RTX 4060 attention validation

1. Keep the main shared-tile implementation as the correctness baseline for the A100/R550 phase.
2. Keep v1-v5 preserved; do not promote v4 or v5 without repeatable gains across representative cases.
3. The first order-alternating current/v5 retest is complete; do not promote v5 unless further independent sessions confirm a robust gain without meaningful regressions.
4. Test a genuinely different SM80 optimization next (e.g. WMMA/tensor-core QK path), only after benchmark controls are trustworthy.
5. Obtain Nsight Compute counter access through the cluster's existing policy/process; do not attempt privileged changes through the runner. The current [actual-cuTile workflow](../.github/workflows/a100-nsight.yml) now runs the correctness suite before attempting two profiles, but it has not been run successfully because the current runner fails the R580+ gate.
6. This historical task is now partly complete: RTX 4060 correctness and controlled benchmarks passed in run #39160827910. The only remaining hard course gate is to obtain approval for the official profiler under project-local .tools and capture two inspectable actual-cuTile profiles, as described in the current results section above.
7. Final report must keep native CUDA surrogate, ordinary CUDA baselines, and actual cuTile results in separate columns/sections.

## 1B. Local hardware inventory received after the initial continuation audit

- GPU: NVIDIA GeForce RTX 4060, compute capability 8.9, 8,188 MiB VRAM (7,925 MiB free at capture).
- OS and driver: Windows 10 Pro x64, build 26300, WDDM; driver 616.92; NVIDIA-SMI reports CUDA UMD 13.4.
- Missing components: `nvcc` and Python are not available; WSL2 and Docker are not installed. This does not require a system Toolkit by itself, because the documented `cuda-tile[tileiras]` and modern CuPy component-wheel route can supply compiler/runtime libraries in a project virtual environment.
- Decision: use native Windows as the first path, install per-user CPython 3.12, then run `scripts/setup_cutile_windows.ps1` and require `CUTILE_SMOKE=PASS`. No real GPU execution was performed by this documentation update.
- A Windows CuPy/NVRTC builtins DLL issue is documented in [NVIDIA cuTile Python issue #57](https://github.com/NVIDIA/cutile-python/issues/57), so installation success and import success are insufficient; the actual vector-add compile/launch/output verification remains the first gate. Detailed profiling can be attempted only after confirming the Nsight Compute binary and counter access.
- New project tooling: `scripts/cutile_smoke.py`, `scripts/setup_cutile_windows.ps1`, `docs/LOCAL_GPU_WINDOWS.md`, and strict `REQUIRE_CUTILE=1` import gates in the two primary cuTile test modules.
