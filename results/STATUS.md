# Current Result Snapshot

## Current main status — 20 October 2026

- **Latest completed full native pass:** [RTX 4060 validation #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), commit `bf5a6233e195c2ced647b4e20b5b2d914da60e85`, artifact [#12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195). Passed actual cuTile smoke; 48 core tests; 3 GQA/MQA/packed-varlen tests; 29 numerical configurations; CLI/NumPy oracle; three controlled benchmark sessions; 36 accuracy-gated latency-hint trials; and two native NCU captures.
- **Current detailed results:** [CUTILE_RTX4060_RESULTS_2026-10-20.md](../docs/CUTILE_RTX4060_RESULTS_2026-10-20.md) lists all 29 tile/workload medians, P90s, session ranges, accuracy maxima, baseline-drift warnings and hint data. Long D=128 / 64×64 measured 0.267059 ms (16.49×); longer-batched D=128 / 64×128 measured 0.915558 ms (18.69×), both versus the intentionally unfused FP32 CuPy baseline.
- **Hint experiment:** medium D=128 / 64×64 with LOAD_LATENCY=10 measured 0.053248 ms vs 0.058342 ms default (8.73% faster with 0.53% ranges for both); longer-batched case did not clear noise-aware threshold. Keep global default at 0 until matched medium hint-10 NCU report is reviewed. Profile run [#39423368441](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423368441) remains queued.
- **PyTorch extension blocker:** CUDA-enabled PyTorch 2.14.0+cu130 fails a plain float32 torch.bmm smoke with CUBLAS_STATUS_INVALID_VALUE before the cuTile bridge is invoked. Do not claim successful autograd, model integration, SDPA backend identification or vendor timing. Isolated CUDA 13.2 wheel diagnostic [#39423804493](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423804493) remains queued and uses a separate project-local venv without overwriting the cuTile venv.
- **Runner follow-up:** repeat native run [#39423190965](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190965) passed setup, smoke and the required core test suite, then stopped during the optional GQA/MQA/packed-varlen step. GitHub shows the job failed while that step remained marked in progress; the job log and artifact are not retrievable, and queued GPU follow-ups have not started. This attempt is not counted as a complete regression pass. Recheck that the local Windows runner/service remains Online.
- **Environment:** Windows x64, GeForce RTX 4060 / SM89, driver 616.92, Python 3.12.10, NumPy 2.5.4, CuPy 14.2.0, cuda-tile 1.6.0, CUDA runtime API 13.2. The Linux A100 remains an ordinary-CUDA/non-cuTile control.
- Workflow state and artifact integrity are tracked in the [execution evidence index](../docs/EXECUTION_EVIDENCE_INDEX.md), full experimental narrative in [Report.md](../Report.md), and current product scope in the [project two-pager](../docs/PROJECT_TWO_PAGER.md).


## Historical baseline verified 2026-10-19 — actual cuTile correctness, benchmark, and Nsight profiling pass

**Primary native run:** [RTX 4060 validation #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223), commit `cd1ec63b210682613ddb29dca3543f0cacc9bacf).  
**Profiler run:** [#39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066).  
**Profiler artifact:** [artifact 12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039), containing four `.ncu-repz` files and exported details/raw CSVs.  
**Hardware:** NVIDIA GeForce RTX 4060; driver 616.92; CC 8.9; 8,188 MiB reported VRAM; Windows x64 build 26300.  
**Environment:** CPython 3.12.10; NumPy 2.5.4; CuPy 14.2.0; cuda-tile 1.6.0; TileIRAS/NVCC/NVVM 13.4.92; Nsight Compute CLI 2026.3.1.0.  
**Verified gates:** genuine smoke PASS; 48 required attention tests PASS with REQUIRE_CUTILE=1; 29 accuracy configurations PASS with zero tolerance violations; CLI/reference PASS; three controlled benchmark sessions PASS; two actual-cuTile NCU profiles and scheduler/warp-state reports captured and inspected.  
**Stable performance highlights:** long D=128, tile 64×64, median 0.267059 ms / P90 0.271626 ms, 16.53× vs unfused CuPy, spread 2.0%, max baseline drift 0.5%. Longer batched D=128, tile 64×128, median 0.923034 ms / P90 0.937656 ms, 18.54×, spread 0.9%, max baseline drift 0.1%.  
**Profiler findings:** the two profiled kernels use 231/255 registers per thread and 65.54/81.92 KiB static shared memory per block; achieved occupancy was 8.35%/8.33%, with shared memory limiting CTA residency to one CTA per SM. Scheduler sections reported no-eligible-warp cycles of 85.83%/85.26%. See [profile analysis](../docs/CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md). The optional stall-sampling metric `smsp__pcsamp_sample_count` was unavailable, so the detailed stall source attribution is limited.  
**Historical-snapshot status:** core required correctness, CLI, controlled-performance, and two-configuration profiling gates passed for this recorded implementation snapshot. Current main has since changed and is being revalidated; do not treat the old result as proof for the newer optional paths. The prior speedup comparison is against the deliberately unfused FP32 CuPy baseline, not vendor-optimized attention.

## Assignment-level closure audit — 2026-10-19

- Full rubric map: [course completion checklist](../docs/COURSE_PROJECT_COMPLETION_CHECKLIST.md).
- Detailed submission narrative: [course technical report](../docs/COURSE_PROJECT_REPORT_2026-10-19.md).
- **Core GPU gates PASS; assignment closure PARTIAL.** The remaining items are a framework/vendor attention reference, clean-cache JIT measurements for long specializations (or an approved exception), the Week 3 second profiler-guided improvement, and owner-confirmed individual contributions. No additional Python packages were installed during this audit.

## Historical runner-discovery failure — before the venv was created

**Workflow:** [RTX 4060 native Windows cuTile validation](https://github.com/xemptynothing/gpu-nvidia/actions/workflows/cutile-rtx4060-docker.yml)  
**Latest attempt:** [run #39154979187](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979187), commit b9189b5d8bec8666a0c22e1747639ba340e0d157.  
**Artifact:** [cutile-rtx4060-experiment, artifact ID 12146591326](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979187/artifacts/12146591326).  
**Hardware preflight:** NVIDIA GeForce RTX 4060; driver 616.92; CC 8.9; 8188 MiB reported memory.  
**Execution result:** failed at Python environment discovery. The only visible interpreter was global CPython 3.12, and its import probe failed with “ModuleNotFoundError: No module named 'numpy'”. The expected .venv-cutile-windows does not exist under the checked-out project root. No packages, profiler tools, or datasets were downloaded or installed. No elevated privileges were used.  
**GPU experiment status:** cuTile smoke, actual attention correctness, CLI/reference, benchmarks and profiler steps all **skipped** because preflight failed. Thus no actual-cuTile attention outcome is available yet.

Run-by-run details, including the earlier pwsh shell failure and Python path attempts, are documented in [docs/PROGRESS.md](../docs/PROGRESS.md) under the 2026-10-19 experiment heading.

**Current blocker:** provide the location of the already-installed Python environment that previously passed the native Windows cuTile smoke test, or put that existing environment under the project root. This workflow will not install or download any dependency.



## Historical continuation status — 2026-10-18 (before actual attention validation)

- Main was `e87d8b9352fdf6783347d5c1e1945172011ebeb9` at the start of the handoff audit; this runner repair advanced main through `bfa8e95bcbc40cff9bc42367f0476cbe222943cc` before this status refresh.
- Move actual-cuTile execution away from the R550.90.07 A100. After inventory, use the user's real local GPU if the NVIDIA support prerequisites are met, otherwise choose another explicitly authorized supported environment. The A100 remains for ordinary-CUDA surrogate testing only.
- Local inventory: RTX 4060 / compute capability 8.9 / driver 616.92 / Windows x64 build 26300, WDDM; 8,188 MiB VRAM. CPython 3.12.10 and cuTile 1.6.0 are installed in the project venv; the genuine vector-add smoke passed. System NVCC remains absent. Docker Desktop with WSL2 backend now reports a working Linux Engine and Compose. The first launcher reached container start but did not confirm runner Online; the old cleanup error masked the original failure.
- **Actions evidence access:** workflow run details and job logs are currently available through the repository connection. CPU/reference run [#75](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39147796099) passed 764 tests / 68 skipped / 0 failed on commit `bfa8e95`; hosted-CPU GPU skips do not count as local cuTile validation.

## Historical pre-validation RTX 4060 inventory — 2026-10-18

| Item | Result |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 |
| Compute capability | 8.9 |
| Driver | 616.92 |
| NVIDIA-SMI CUDA UMD | 13.4 (driver-reported support level, not Toolkit proof) |
| VRAM | 8,188 MiB total; 7,925 MiB free at capture |
| OS | Windows 10 Pro x64, build 26300; WDDM |
| Python | CPython 3.12.10 in project virtual environment |
| System NVCC/Toolkit | `nvcc` not found |
| WSL2 / Docker | WSL2 backend active; Docker Desktop 4.95.0, Engine 29.8.2, Compose 5.6.0; the launcher reached container start after its GPU probe gate |
| Initial route | Native Windows x64 venv; see [local GPU guide](../docs/LOCAL_GPU_WINDOWS.md) |
| Actual cuTile smoke | **PASS** — genuine vector-add kernel; synchronization and exact output check passed |
| Attention correctness / performance / profiler | **NOT RUN YET** |

The GPU/driver/OS meet NVIDIA's current Quickstart prerequisites on paper (Windows x86_64, CC 8.x–12.x, R580+). The genuine cuTile smoke passed, but attention correctness remains open. Docker Desktop/WSL2 are running and Compose is available. The first launcher started the container but did not confirm GitHub Online; the initial cleanup errored on missing Compose interpolation variables. The revised persistent launcher takes an owner-supplied short-lived registration token and refuses to pull the GPU-probe image if it is absent locally.

## Actual self-hosted A100 runner inventory

Verified by successful A100 driver-access audit run on the self-hosted runner:

- Runner: `gpu-project`
- Machine: `scn21-mn`
- Runner version: 2.337.0
- Host OS: Ubuntu 22.04 family, kernel 5.15.0-1062-nvidia, glibc 2.35
- Host architecture: x86_64
- System Python: 3.14.6
- Physical host: 8 x NVIDIA A100-SXM4-40GB + 6 x second-generation NVSwitches
- CUDA-visible to this runner: 2 x full A100-SXM4-40GB via `CUDA_VISIBLE_DEVICES=0,1`
- Selected GPUs: compute capability 8.0, 40,960 MiB each, MIG disabled
- Selected GPUs: PCIe Gen4 x16
- Selected GPUs: 12 NVLink links, each reporting 25 GB/s
- Selected GPUs: ECC enabled, no volatile DRAM/SRAM correctable or uncorrectable errors at audit time
- Selected GPUs: no active processes, ~1 MiB used, 29 C, ~55-56 W idle power, 400 W power limit
- GPU reset status: no reset required
- GPU virtualization mode: none
- NVSwitch/Fabric Manager: installed and active
- System CUDA Toolkit: 12.4.131
- GCC/G++: 11.4.0
- Nsight Compute: 2024.1.1.0
- Nsight Systems: 2023.4.4.54
- DCGM: installed; sees all 8 physical GPUs and 6 NVSwitches
- Runner account: non-root; sudo is not passwordless

## Historical validation state before the actual attention run

| Area | Result |
|---|---|
| CPU/reference CI | PASS: run [#75](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39147796099), 764 passed / 68 skipped / 0 failed (commit bfa8e95; GPU tests skip on hosted CPU) |
| Self-hosted A100 runner routing | PASS |
| Actual A100 hardware inventory | PASS; exhaustive read-only audit |
| Basic native CUDA execution | PASS |
| Native CUDA RawKernel correctness | PASS: native run #44, 133 passed / 1 skipped |
| Broad non-cuTile regression | PASS: [run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082), 907 passed / 52 skipped |
| Shared-tile v4 experiment | PASS correctness: 30/30; not promoted |
| Shared-tile v5 experiment | PASS correctness: 30/30; three paired/order-alternating repeats; not promoted |
| Local cuTile runtime smoke | PASS on RTX 4060 / driver 616.92; attention correctness not yet run |
| CUDA 13.4 forward-compatibility experiment | Failed: insufficient-driver / CUDA error 803 |
| CUDA 13.0 forward-compatibility attempt | No verified successful end-to-end cuTile GPU run |
| Nsight Compute binary presence | PASS: 2024.1.1 |
| Nsight Compute performance-counter access | BLOCKED: `ERR_NVGPUCTRPERM`; no kernels profiled |
| PTXAS resource reporting | PASS: v1/v2/v3/current/v4/v5 compiled; no reported spills; v5 cubin 15,520 B |
| Repeated current-vs-v5 benchmark | PASS: [run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163), three paired repeats/case/tile; v5 not promoted due to mixed/noisy gains |
| Nsight Systems timeline (surrogate) | PASS: [run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950); two `.nsys-rep` traces + CUDA API/kernel summaries uploaded ([artifact](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954)) |
| Native-CUDA benchmark artifacts | PASS: CUDA-event ladder plus repeated paired current-vs-v5 artifacts uploaded to Actions |
| Actual cuTile benchmark/profiler artifacts | NOT AVAILABLE; only the vector-add smoke has executed, not the attention kernel |
| Fabric Manager | PRESENT and ACTIVE |

## Historical A100 interpretation

The A100 hardware itself is suitable for the project. The runner is attached to a
larger HGX/NVSwitch host but currently exposes two full, non-MIG A100s.

The native CUDA surrogate now executes correctly; the early NVRTC header-discovery issue was avoided by compiling RawKernels with NVCC and matching CUDA 12.4 headers. Remaining blockers are:
1. actual cuTile needs a supported R580+ driver/runtime;
2. Nsight Compute counter collection is denied to the runner account (`ERR_NVGPUCTRPERM`);
3. v4/v5 performance evidence is mixed and needs more controlled repeatability before promotion.

For the full approach matrix, test counts, timing evidence, and next decisions, see
[docs/EXPERIMENT_REPORT.md](../docs/EXPERIMENT_REPORT.md). Ordinary-CUDA timings
must not be reported as cuTile timings.

## Historical runner repair update — 2026-10-18 (before the successful run)

- Launcher/token flow and persistent runner changes are committed in main; the latest CPU/reference structural CI for the contract update passed 764 / 68 skipped / 0 failed.
- No statement is made that the persistent local runner is Online or that the cuTile GitHub Actions workflow passed. Re-run with an owner-created registration token and inspect the container logs / GitHub runner list.
- The user-reported 34 local targeted GPU tests passed in 41.91 seconds, but this status does not substitute for the full workflow evidence, a verified `REQUIRE_CUTILE=1` run, the CLI contract, performance measurements, or profiles.

## Optional-objective source changes — native validation pending

The source now includes optional GQA/MQA and packed-varlen wrappers/references/tests, a noise-aware tile autotuner, a PyTorch autograd/model integration path and a cuTile DRAM load-latency hint sweep. CPU/reference CI validates only pure NumPy semantics, recommendation logic and Python syntax; GPU tests still need a successful run on the RTX 4060. A screenshot helper attempts real Nsight GUI captures and does not count blank captures. Do not mark any of these optional GPU features complete until reports, numerical output and repeated timing data have been inspected. The previously successful core native run applies to its original source baseline; a clean full regression is required after the added KV_GROUPS/LOAD_LATENCY compile-time parameters.
