# Nsight Compute results

The existing reports were captured on the Windows RTX 4060. They are from the earlier validated source state, before the later kernel compaction. Reprofile the current source before using these numbers as current.

## Captured configurations

| Metric | medium_d128, 64x64 | longer_batched_d128, 64x128 |
|---|---:|---:|
| Shape (B,H,N,D) | (1,8,512,128) | (2,8,2048,128) |
| Profiled duration | 52.90 us | 984.26 us |
| Threads per block | 128 | 128 |
| Registers per thread | 231 | 255 |
| Static shared memory per block | 65.54 KiB | 81.92 KiB |
| Achieved occupancy | 8.35% | 8.33% |
| SM throughput vs device peak | 18.81% | 30.27% |
| DRAM throughput vs device peak | 22.88% | 11.89% |
| Scheduler cycles with no eligible warp | 85.83% | 85.26% |

Reports and CSV exports: [artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039). The core GPU run and benchmark record is [#39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438).

## What the numbers suggest

- Shared memory limits resident blocks in both profiles. Register use is also high.
- Few eligible warps and many cycles without an eligible warp suggest limited latency hiding.
- Neither case is close to peak SM throughput or DRAM bandwidth.
- Low DRAM throughput alone does not prove a memory bottleneck. Check L1/L2, shared memory and warp states together.
- Nsight's estimated local-speedup figures are heuristics, not measured gains. The detailed PC-sampling metric was unavailable, so source-level stall attribution is limited.

## Next profile run

For each selected workload, compare 64x64 with 64x128 using the same input. Save Summary, Details, Raw, Occupancy, Memory, Scheduler and Warp State views. Keep the .ncu-rep/.ncu-repz report files beside the PNGs.

Pick one change from the evidence. Run correctness first, then repeated unprofiled timing, then a matching before/after profile. Keep the change only if the results support it.
