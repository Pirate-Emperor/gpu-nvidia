# Current Result Snapshot

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
| CPU/reference CI | PASS; latest run completed successfully |
| Self-hosted A100 runner routing | PASS |
| Actual A100 hardware inventory | PASS; exhaustive read-only audit |
| Basic native CUDA execution | PASS |
| Native CUDA RawKernel correctness | BLOCKED/FAILED at JIT on initial attempt because NVRTC could not find standard headers |
| Matching CUDA 12.4 runtime-header package | Added to native CUDA workflow; follow-up run must validate |
| Current cuTile runtime | BLOCKED by R550.90.07 |
| cuTile forward-compatibility experiment | CUDA error 803 |
| Nsight binary presence | PASS |
| Actual Nsight performance-counter access | NOT YET PROVEN; dedicated smoke test added |
| Fabric Manager | PRESENT and ACTIVE |
| Benchmark/profiler raw results | Pending supported execution |

## Interpretation

The A100 hardware itself is suitable for the project. The runner is attached to a
larger HGX/NVSwitch host but currently exposes two full, non-MIG A100s.

The remaining issues are software/runtime rather than GPU capability:
1. the host NVIDIA driver is R550.90.07 while current cuTile requires R580+;
2. the first CuPy RawKernel attempt exposed a CUDA 12.4 NVRTC header-discovery issue;
3. Nsight performance-counter permission still needs a real profiler smoke test.

No cuTile correctness or performance number is considered valid until the real
cuTile kernel executes on the supported environment.
