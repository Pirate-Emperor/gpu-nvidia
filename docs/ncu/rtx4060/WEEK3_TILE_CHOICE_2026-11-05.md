# Week 3 tile-choice optimization — longer-batched D=128

**Date:** 5 November 2026  
**Platform:** Native Windows NVIDIA GeForce RTX 4060, SM 8.9  
**Source revision under test:** expanded current-source native validation run [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189)  
**Matched profiler run:** [#42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947)  
**Artifact:** [cutile-rtx4060-ncu-profiles, #13440087603](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947/artifacts/13440087603)

## Why this is the second Week 3 improvement

The course-project brief requires a second, distinct profiler-guided improvement in Week 3. The Week 2 optimization is the medium-D=128 LOAD_LATENCY=10 specialization. This note records a separate tile-size decision for the longer-batched D=128 case; it is not a re-labelling of that first hint experiment.

Workload: B=2, H=8, N=2048, D=128, Dv=128. The variable was key-tile width: Tq=64, Tk=64 versus Tk=128. Both configurations pass the cuTile numerical matrix.

## Stable three-session benchmark

The expanded current-source matrix in run #42075683189 used three order-varied sessions, ten warmups, 30 CUDA-event samples per tile/session and five launches per sample. The speedup denominator is the deliberately unfused FP32 CuPy baseline.

| Tile | Fused median | Session range | Max baseline-bracket drift | Speedup vs unfused FP32 CuPy |
|---|---:|---:|---:|---:|
| 64×64 | 0.936346 ms | 0.525% | 0.055% | 18.29× |
| 64×128 | 0.921395 ms | 0.456% | 0.055% | 18.60× |

Both configurations pass the predeclared ≤5% session-spread and ≤5% baseline-drift gates. The 64×128 tile is **1.60% faster** by median in this workload.

## Matched Nsight Compute comparison

The same local RTX 4060, same longer-batched input and same profile command were used for both tile choices. Each choice has a basic resource/throughput report and a separate scheduler/warp-state report. The raw native .ncu-repz reports and Details/Raw CSVs are retained in the Actions artifact linked above.

| Metric | 64×64 | 64×128 |
|---|---:|---:|
| NCU kernel duration | 997.73 µs | 990.98 µs |
| Memory throughput | 19.65% | 20.21% |
| DRAM throughput | 11.67% | 11.81% |
| SM compute throughput | 29.03% | 30.23% |
| Registers per thread | 227 | 255 |
| Static shared memory per CTA | 65.54 KiB | 81.92 KiB |
| Achieved occupancy | 8.33% | 8.33% |
| Eligible warps per scheduler | 0.16 | 0.15 |
| No-eligible scheduler cycles | 84.48% | 85.24% |
| Warp cycles per issued instruction | 6.44 | 6.80 |

The matched instrumented duration is 0.68% lower for 64×128. Memory and DRAM throughput rise slightly, and SM throughput also rises slightly. However, the larger tile costs 28 more registers per thread and 16.38 KiB more static shared memory per CTA. Occupancy remains 8.33%; eligible warps per scheduler fall from 0.16 to 0.15, and no-eligible cycles rise from 84.48% to 85.24%.

## Decision and limits

**Accept 64×128 only for longer-batched D=128 / N=2048.** The independent stable benchmark and matched NCU profile agree on a modest benefit, satisfying the second Week 3 tuning loop. Keep 64×64 as the global tile default. Do not describe this as a broad kernel-wide gain.

Two isolated paired retries are deliberately not used as headline results:
- [Run #42073544473](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42073544473): the 64×64 comparison exceeded the 5% session-spread gate (5.22%); the experiment stopped before profiling.
- [Run #42073654984](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42073654984): both tile choices exceeded the 5% spread gate. Its apparent speed difference is retained as a variance/negative result, not a stable performance claim.

The remaining bottleneck is still CTA resource pressure and latency hiding. Occupancy did not improve, eligible-warp count is very small, and DRAM throughput is only about 12% of peak. The next hypothesis is to reduce shared-memory/register footprint or increase eligible warps per SM, then repeat correctness, stable benchmarking and matched NCU.

## Related validation

- Expanded native RTX 4060 validation: [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189), 57 cuTile correctness tests passed, 29/29 numerical rows passed, zero tolerance violations.
- Matched tile-choice NCU and full profile artifact: [#42076178947](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947), [artifact #13440087603](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42076178947/artifacts/13440087603).
