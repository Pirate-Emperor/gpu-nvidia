# Current Result Snapshot

## Continuation status — 2026-10-18

- Main was `e87d8b9352fdf6783347d5c1e1945172011ebeb9` at the start of the handoff audit; this runner repair advanced main through `bfa8e95bcbc40cff9bc42367f0476cbe222943cc` before this status refresh.
- Move actual-cuTile execution away from the R550.90.07 A100. After inventory, use the user's real local GPU if the NVIDIA support prerequisites are met, otherwise choose another explicitly authorized supported environment. The A100 remains for ordinary-CUDA surrogate testing only.
- Local inventory: RTX 4060 / compute capability 8.9 / driver 616.92 / Windows x64 build 26300, WDDM; 8,188 MiB VRAM. CPython 3.12.10 and cuTile 1.6.0 are installed in the project venv; the genuine vector-add smoke passed. System NVCC remains absent. Docker Desktop with WSL2 backend now reports a working Linux Engine and Compose. The first launcher reached container start but did not confirm runner Online; the old cleanup error masked the original failure.
- **Actions evidence access:** workflow run details and job logs are currently available through the repository connection. CPU/reference run [#75](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39147796099) passed 764 tests / 68 skipped / 0 failed on commit `bfa8e95`; hosted-CPU GPU skips do not count as local cuTile validation.

## Local actual-cuTile candidate — RTX 4060 (inventory supplied 2026-10-18)

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

## Validation state

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

## Interpretation

The A100 hardware itself is suitable for the project. The runner is attached to a
larger HGX/NVSwitch host but currently exposes two full, non-MIG A100s.

The native CUDA surrogate now executes correctly; the early NVRTC header-discovery issue was avoided by compiling RawKernels with NVCC and matching CUDA 12.4 headers. Remaining blockers are:
1. actual cuTile needs a supported R580+ driver/runtime;
2. Nsight Compute counter collection is denied to the runner account (`ERR_NVGPUCTRPERM`);
3. v4/v5 performance evidence is mixed and needs more controlled repeatability before promotion.

For the full approach matrix, test counts, timing evidence, and next decisions, see
[docs/EXPERIMENT_REPORT.md](../docs/EXPERIMENT_REPORT.md). Ordinary-CUDA timings
must not be reported as cuTile timings.

## Runner repair update — 2026-10-18

- Launcher/token flow and persistent runner changes are committed in main; the latest CPU/reference structural CI for the contract update passed 764 / 68 skipped / 0 failed.
- No statement is made that the persistent local runner is Online or that the cuTile GitHub Actions workflow passed. Re-run with an owner-created registration token and inspect the container logs / GitHub runner list.
- The user-reported 34 local targeted GPU tests passed in 41.91 seconds, but this status does not substitute for the full workflow evidence, a verified `REQUIRE_CUTILE=1` run, the CLI contract, performance measurements, or profiles.
