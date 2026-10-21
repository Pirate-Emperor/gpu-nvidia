# Runtime Tile Sweep: Mentor Demonstration

## Purpose

The normal `scripts/mentor_demo.ps1` command does not stop at a PASS marker. After checking the GPU, running a genuine cuTile smoke kernel, executing the required attention tests and verifying the command-line output against the independent NumPy oracle, it runs a presentation-sized runtime tile sweep and prints the timing table.

Run from the project root:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_demo.ps1
```

Use `-SkipTileSweep` only for a quicker smoke/test/CLI check. Use `-WithProfile` to add a live NCU attempt after the runtime comparison.

## What gets compared

Two representative workloads are held fixed while the query/key tile sizes change:

| Case | Shape (B,H,N,D) | Purpose |
|---|---|---|
| `medium_d128` | (1,8,512,128) | Medium sequence and eight heads |
| `longer_batched_d128` | (2,8,2048,128) | Longer sequence and two-item batch |

Candidate tiles: `32x64`, `64x64`, `64x128`, `128x64`, `128x128`. Both N values are divisible by the candidates, so every comparison is legal under the current fixed-tile contract.

Timing protocol for each case/tile pair: 10 warmups, 30 CUDA-event samples, five identical launches inside each sample. The event elapsed time is divided by five and then the median/P90 are calculated over the 30 per-invocation values. The demo has one session; the full project benchmark uses three order-rotated sessions across the complete required matrix.

## Console fields

| Printed field | Definition | Decision use |
|---|---|---|
| `MEDIAN ms` | Median per-launch steady-state CUDA-event latency | Main latency comparison within one case; lower is faster |
| `P90 ms` | 90th percentile sample latency | Shows the slow tail; compare with median to discuss variation |
| `BASE ms` | Average of baseline median timings taken before and after the tile sweep | Reference is the deliberately unfused FP32 CuPy baseline, not optimized SDPA |
| `SPEEDUP` | `BASE ms / MEDIAN ms` | Only a speedup against the unfused FP32 CuPy baseline |
| `DRIFT %` | Absolute before/after baseline median change divided by their average | High drift warns of temperature/clock/background-state movement; repeat instead of trusting a small gap |
| `FIRST CALL` | Wall time for first use of a kernel specialization in this process | May be a compiler-cache hit; do not call it clean-cache JIT unless cache state was deliberately controlled |

The output also prints the lowest-median tile per case, explicitly labelled as the best result **for this measured sweep only**. The raw data are preserved under `.tmp\logs` as timestamped `mentor_benchmark_*.json` and `mentor_benchmark_*.csv` files. JSON includes GPU/runtime metadata, seed, warmups, samples, repeats, baseline-before/after measurements, drift, JIT/first-call field, medians, P90s, speedups and output-byte throughput.

## Why the comparison is interesting

The preserved three-session aggregate (historical reference, not a prediction for each fresh run) recorded:

| Workload | Tile | Median | P90 | Speedup vs unfused FP32 CuPy |
|---|---:|---:|---:|---:|
| medium D=128 | 32x64 | 0.069632 ms | 0.070659 ms | 5.62x |
| medium D=128 | 64x64 | 0.058368 ms | 0.058814 ms | 6.70x |
| medium D=128 | 64x128 | 0.058368 ms | 0.059044 ms | 6.70x |
| medium D=128 | 128x64 | 0.079053 ms | 0.079258 ms | 4.94x |
| medium D=128 | 128x128 | 0.107315 ms | 0.107950 ms | 3.64x |
| longer batched D=128 | 32x64 | 1.142989 ms | 1.146900 ms | 14.97x |
| longer batched D=128 | 64x64 | 0.932966 ms | 0.936468 ms | 18.33x |
| longer batched D=128 | 64x128 | 0.915558 ms | 0.921457 ms | 18.69x |
| longer batched D=128 | 128x64 | 1.017139 ms | 1.020314 ms | 16.83x |
| longer batched D=128 | 128x128 | 1.285120 ms | 1.289237 ms | 13.32x |

Interpretation:
- The medium D=128 aggregate has a median tie between 64x64 and 64x128; keep 64x64 as the general default rather than overreacting to a tie.
- In the longer batched case, 64x128 was about 1.9% faster than 64x64 in the aggregate, a plausible specialization worth comparing in repeated sessions.
- Larger tiles are not automatically faster. They may reduce key-tile loop overhead but increase the live tile footprint and pressure registers/shared memory. Previous NCU profiles show why timing needs profiler context.
- The full benchmark's three sessions rotate/reverse tile order. A single live demo run is illustrative; a small win should not be promoted without repeat-session evidence and reasonable baseline drift.

## What to say

> “Correctness is the first gate, then I want to understand performance. I hold each workload fixed and change only the tile configuration. Thirty CUDA-event samples each average five launches after ten warmups. I compare median, P90 and baseline drift rather than choosing the smallest isolated number. On the recorded aggregate, 64x64 is a good general default, while 64x128 is about 1.9% faster on the longer batched case. That is workload-dependent tuning, not a claim that bigger tiles always win.”

Then transition to Nsight Compute:

> “The timing tells me which configuration was faster in this measurement, not why. NCU exposes the resource footprint, occupancy, throughput and eligible-warps behavior. Those metrics help select the next hypothesis. I still rerun correctness and controlled timings after any change; NCU's estimated local speedup is a heuristic, not a measured gain.”

## Scope and caution

The measured baseline is an intentionally unfused FP32 CuPy implementation that materializes scores/probabilities. It is not FlashAttention or optimized PyTorch SDPA. The framework/vendor attention comparison remains separately open. Existing clean-cache JIT timing for some long specializations is also incomplete because first-call values may represent cache hits.