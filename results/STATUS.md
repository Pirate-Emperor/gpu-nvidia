# Current Result Snapshot

## Continuation status — 2026-10-18

- At the start of this continuation, `main` was `e87d8b9352fdf6783347d5c1e1945172011ebeb9`, matching the handoff SHA.
- Move actual-cuTile execution away from the R550.90.07 A100. After inventory, use the user's real local GPU if the NVIDIA support prerequisites are met, otherwise choose another explicitly authorized supported environment. The A100 remains for ordinary-CUDA surrogate testing only.
- Local GPU inventory and runner setup are pending. Do not select Windows-native versus WSL2/Linux, register a runner, or claim OS isolation before the actual GPU/OS/driver/toolchain is known and the boundary is tested.
- **Actions evidence access note (2026-10-18 continuation):** the connected Actions run list returned zero visible runs; direct run-detail, job and artifact requests for handoff IDs (including #343, #52, #340, benchmark #4 and Nsight Systems #2) returned 404; the current handoff commit had no visible combined status. This does not prove historical runs are absent. Historical counts below remain from the prior handoff and were not independently re-audited through this connection.

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
| CPU/reference CI | PASS: latest verified run [#340](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056390931), 763 passed / 68 skipped (pre-handoff main SHA afe530b) |
| Self-hosted A100 runner routing | PASS |
| Actual A100 hardware inventory | PASS; exhaustive read-only audit |
| Basic native CUDA execution | PASS |
| Native CUDA RawKernel correctness | PASS: native run #44, 133 passed / 1 skipped |
| Broad non-cuTile regression | PASS: [run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082), 907 passed / 52 skipped |
| Shared-tile v4 experiment | PASS correctness: 30/30; not promoted |
| Shared-tile v5 experiment | PASS correctness: 30/30; three paired/order-alternating repeats; not promoted |
| Current cuTile runtime | BLOCKED by R550.90.07; R580+ gate |
| CUDA 13.4 forward-compatibility experiment | Failed: insufficient-driver / CUDA error 803 |
| CUDA 13.0 forward-compatibility attempt | No verified successful end-to-end cuTile GPU run |
| Nsight Compute binary presence | PASS: 2024.1.1 |
| Nsight Compute performance-counter access | BLOCKED: `ERR_NVGPUCTRPERM`; no kernels profiled |
| PTXAS resource reporting | PASS: v1/v2/v3/current/v4/v5 compiled; no reported spills; v5 cubin 15,520 B |
| Repeated current-vs-v5 benchmark | PASS: [run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163), three paired repeats/case/tile; v5 not promoted due to mixed/noisy gains |
| Nsight Systems timeline (surrogate) | PASS: [run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950); two `.nsys-rep` traces + CUDA API/kernel summaries uploaded ([artifact](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954)) |
| Native-CUDA benchmark artifacts | PASS: CUDA-event ladder plus repeated paired current-vs-v5 artifacts uploaded to Actions |
| Actual cuTile benchmark/profiler artifacts | NOT AVAILABLE; real cuTile kernel not executed |
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
