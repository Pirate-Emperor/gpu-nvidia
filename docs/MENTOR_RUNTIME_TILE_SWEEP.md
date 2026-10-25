# Runtime tile sweep

Run from the repository root:

~~~powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_demo.ps1
~~~

The demo checks the GPU, launches a small cuTile kernel, runs the attention tests and checks a saved CLI result. It then compares five tile shapes on two fixed workloads.

| Case | Shape (B,H,N,D) |
|---|---|
| medium_d128 | (1,8,512,128) |
| longer_batched_d128 | (2,8,2048,128) |

Tiles: 32x64, 64x64, 64x128, 128x64 and 128x128. Each case/tile pair uses 10 warmups and 30 CUDA-event samples, with five launches per sample. The quick demo runs one session; the full benchmark uses three sessions with rotated tile order.

The output shows median and P90 latency, the unfused FP32 CuPy baseline, speedup versus that baseline, timing variation, and first-call time. First-call time is not a clean JIT measurement unless the cache state is controlled.

Do not pick a tile from a single result. Use repeated timings, compare the same workload, and check the baseline variation. The full measurements are in Report.md and the dated RTX 4060 results files.
