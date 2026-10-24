# Native Windows RTX 4060 cuTile Results — 20 October 2026

**Primary evidence:** [completed native validation run #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), source commit `bf5a6233e195c2ced647b4e20b5b2d914da60e85`. **Raw artifact:** [#12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195), 194,711 bytes, SHA-256 `e3eaf4ae1e88d2547d2034dfe333db86b70f69e2f2d6c71b0d67b7c8bdde0c32`.

This page supersedes the 19 October result snapshot for the measurements below. It records actual NVIDIA cuTile execution on the Windows RTX 4060; the Linux A100 ordinary-CUDA runner is not used as evidence for these numbers. The kernel was compacted after the recorded run; rerun current main before treating these results as current-source validation.

## Outcome

- Genuine cuTile smoke kernel: passed; 4,096 outputs checked and stream synchronization verified.
- Required core GPU tests with `REQUIRE_CUTILE=1`: **48 passed, 0 failed**.
- Selected optional GPU tests: **3 passed** for GQA/MQA and packed-variable-length behavior.
- Independent FP32 NumPy accuracy matrix: **29 legal workload/tile combinations, 0 tolerance violations**.
- CLI-generated output: shape/dtype/finiteness and independent NumPy oracle check passed.
- Performance: three controlled sessions, 29 workload/tile pairs, 10 warmups, 30 CUDA-event samples, five launches per sample, order rotated/reversed between sessions.
- Native Nsight Compute: two reports were produced by this run. More detailed report exports/screenshots are in the dedicated profiler artifact referenced in the main report and evidence index.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, compute capability 8.9 / SM89 |
| Driver | 616.92 |
| OS | Windows 11, build 26300 (native Windows; not WSL) |
| Python | CPython 3.12.10 |
| NumPy | 2.5.4 |
| CuPy | 14.2.0 |
| NVIDIA cuTile Python | cuda-tile 1.6.0 |
| CUDA runtime API reported by CuPy | 13.2 (13020) |
| CUDA driver API version reported by CuPy | 13.4 (13040) |
| CUDA TileIRAS / NVCC / NVVM packages | 13.4.92 series |
| Toolkit Python package | cuda-toolkit 13.4.3 |
| Nsight Compute CLI | 2026.3.1.0, build 38829034 |
| Tensor contract | FP16 Q/K/V and FP16 output; FP32 accumulation and online-softmax state |

## Three-session performance matrix

The table lists across-session medians from the exact benchmark summary JSON in the artifact. P90 is the median of each configuration's three session P90 values; session range is the min-to-max spread of session-level median latencies divided by the across-session median. “Speedup” is against the deliberately unfused FP32 CuPy baseline that materializes score/probability intermediates. It is **not** speedup against FlashAttention, optimized PyTorch SDPA, or another fused vendor kernel.

| Workload | Shape (B,H,N,D) | Tile (TQ×TK) | Median ms | P90 ms | Speedup | Session range | First-call ms by session |
|---|---:|---:|---:|---:|---:|---:|---|
| long_d128 | 1,16,1024,128 | 32×64 | 0.334950 | 0.337736 | 13.11× | 0.06% | —, —, — |
| long_d128 | 1,16,1024,128 | 64×64 | 0.267059 | 0.269128 | 16.49× | 0.92% | —, —, — |
| long_d128 | 1,16,1024,128 | 64×128 | 0.270848 | 0.273449 | 16.19× | 1.13% | —, —, — |
| long_d128 | 1,16,1024,128 | 128×64 | 0.316109 | 0.319324 | 13.89× | 0.78% | —, —, — |
| long_d128 | 1,16,1024,128 | 128×128 | 0.411853 | 0.415846 | 10.66× | 0.79% | —, —, — |
| longer_batched_d128 | 2,8,2048,128 | 32×64 | 1.142989 | 1.146900 | 14.97× | 0.31% | —, —, — |
| longer_batched_d128 | 2,8,2048,128 | 64×64 | 0.932966 | 0.936468 | 18.33× | 0.33% | —, —, — |
| longer_batched_d128 | 2,8,2048,128 | 64×128 | 0.915558 | 0.921457 | 18.69× | 0.50% | —, —, — |
| longer_batched_d128 | 2,8,2048,128 | 128×64 | 1.017139 | 1.020314 | 16.83× | 0.23% | —, —, — |
| longer_batched_d128 | 2,8,2048,128 | 128×128 | 1.285120 | 1.289237 | 13.32× | 0.32% | —, —, — |
| medium_d128 | 1,8,512,128 | 32×64 | 0.069632 | 0.070659 | 5.62× | 5.00% | 29.4, 27.3, 27.1 |
| medium_d128 | 1,8,512,128 | 64×64 | 0.058368 | 0.058814 | 6.70× | 9.82% | 26.2, 27.1, 28.2 |
| medium_d128 | 1,8,512,128 | 64×128 | 0.058368 | 0.059044 | 6.70× | 2.46% | 86.2, 80.3, 28.0 |
| medium_d128 | 1,8,512,128 | 128×64 | 0.079053 | 0.079258 | 4.94× | 5.31% | 27.6, 28.5, 81.9 |
| medium_d128 | 1,8,512,128 | 128×128 | 0.107315 | 0.107950 | 3.64× | 3.05% | 29.4, 32.9, 30.0 |
| medium_d64 | 1,8,256,64 | 32×64 | 0.020070 | 0.021238 | 14.00× | 95.92% | —, —, — |
| medium_d64 | 1,8,256,64 | 64×64 | 0.017171 | 0.017833 | 16.25× | 109.73% | —, —, — |
| medium_d64 | 1,8,256,64 | 64×128 | 0.019456 | 0.020480 | 14.40× | 90.05% | —, —, — |
| medium_d64 | 1,8,256,64 | 128×64 | 0.019430 | 0.022733 | 14.22× | 120.16% | —, —, — |
| medium_d64 | 1,8,256,64 | 128×128 | 0.024944 | 0.026010 | 11.15× | 76.36% | —, —, — |
| short_d32 | 1,1,64,32 | 32×64 | 0.015770 | 0.018473 | 19.10× | 2.09% | 1555.0, 27.2, 27.2 |
| short_d32 | 1,1,64,32 | 64×64 | 0.015974 | 0.018125 | 18.73× | 0.64% | 27.8, 1562.3, 1567.4 |
| short_d64 | 1,1,64,64 | 32×64 | 0.016074 | 0.018186 | 18.25× | 5.20% | 26.9, 26.7, 27.8 |
| short_d64 | 1,1,64,64 | 64×64 | 0.016282 | 0.018862 | 18.33× | 1.36% | 26.9, 27.2, 27.1 |
| short_multihead_d64 | 1,4,128,64 | 32×64 | 0.016074 | 0.018810 | 16.96× | 3.70% | —, —, — |
| short_multihead_d64 | 1,4,128,64 | 64×64 | 0.016179 | 0.018452 | 17.15× | 1.29% | —, —, — |
| short_multihead_d64 | 1,4,128,64 | 64×128 | 0.016566 | 0.019313 | 16.96× | 3.71% | 27.2, 27.8, 27.2 |
| short_multihead_d64 | 1,4,128,64 | 128×64 | 0.016794 | 0.018043 | 16.69× | 3.66% | 26.7, 28.5, 27.9 |
| short_multihead_d64 | 1,4,128,64 | 128×128 | 0.018842 | 0.020910 | 14.88× | 4.35% | 26.9, 29.6, 26.8 |

### Interpretation

- **Long D=128:** tile 64×64 is the best measured median (0.267059 ms, 16.49×) and 64×128 is close but slower (0.270848 ms, about 1.42% slower). Keep 64×64 as the general default.
- **Longer batched D=128:** 64×128 is the best measured tile (0.915558 ms, 18.69×) and is about 1.86% faster than 64×64 (0.932966 ms). This is the best-supported workload-specific specialization in the matrix.
- **Medium D=128:** 64×64 and 64×128 tie at 0.058368 ms across sessions, but 64×64 has a 9.82% spread and 64×128 a 2.46% spread. The standalone hint experiment below is more reliable for this workload’s scheduling hypothesis.
- **Medium D=64 and short cases:** relative variation is high in some cases. Do not headline their ratios as precise gains; use the longer workloads for the clearest performance story.

## Independent numerical accuracy

All 29 case/tile combinations passed the mixed tolerance `abs(error) <= 0.02 + 0.02 * abs(reference)`. Summary maxima from the independent FP32 NumPy oracle:

| Metric | Worst observed | Case / tile |
|---|---:|---|
| Maximum absolute error | 0.0009765625 | short_d32 / 32×64 |
| Largest P99 absolute error | 0.00048828125 | short_d32 / 32×64 |
| Largest RMSE | 0.000086860741 | short_d32 / 32×64 |
| Maximum relative error for |reference| ≥ 0.01 | 2.2459% | medium_d128 / 32×64 |
| Maximum tolerance-envelope ratio | 2.4367% | long_d128 / 32×64 |
| Tolerance violations across all configurations | 0 | 29/29 pass |

Do not confuse the maximum tolerance-envelope ratio (2.44% of allowed tolerance) with a 2.44% output error.

## DRAM load-latency hint sweep

Run by the same native artifact and saved at `results/cutile-load-latency-hints.json` / `.csv`. All 36 hint/session trials passed the independent NumPy accuracy check before timing. Each candidate was measured with 10 warmups, 30 CUDA-event samples, five launches per sample and three order-rotated sessions. The decision rule requires a candidate to beat both the 1% minimum gain and half the combined default/candidate session spread.

| Workload | Hint | Session medians ms | Median ms | Gain vs hint 0 | Three-session range | Decision |
|---|---:|---|---:|---:|---:|---|
| Medium D=128 / 64×64 | 0 | 0.058061, 0.058342, 0.058368 | 0.058342 | baseline | 0.53% | default |
| Medium D=128 / 64×64 | 2 | 0.055296, 0.054886, 0.054682 | 0.054886 | 5.92% | 1.12% | passes accuracy; improvement is not needed as default over hint 10 |
| Medium D=128 / 64×64 | 4 | 0.058368, 0.058368, 0.058368 | 0.058368 | -0.04% | 0.00% | no gain |
| Medium D=128 / 64×64 | 6 | 0.058368, 0.058448, 0.058368 | 0.058368 | -0.04% | 0.14% | no gain |
| Medium D=128 / 64×64 | 8 | 0.056525, 0.056397, 0.056515 | 0.056515 | 3.13% | 0.23% | passes accuracy; slower than hint 10 |
| Medium D=128 / 64×64 | 10 | 0.053325, 0.053248, 0.053043 | 0.053248 | 8.73% | 0.53% | **credible timing candidate; matched NCU review pending** |
| Longer batched D=128 / 64×128 | 0 | 0.992256, 0.920883, 0.922010 | 0.922010 | baseline | 7.74% | retain default |
| Longer batched D=128 / 64×128 | 2 | 0.957542, 0.920576, 0.920576 | 0.920576 | 0.16% | 4.02% | below noise-aware required gain |
| Longer batched D=128 / 64×128 | 4 | 0.922214, 0.920883, 0.920576 | 0.920883 | 0.12% | 0.18% | below required 5.88% threshold |
| Longer batched D=128 / 64×128 | 6 | 0.921498, 0.920576, 0.920678 | 0.920678 | 0.14% | 0.10% | below required 5.88% threshold |
| Longer batched D=128 / 64×128 | 8 | 0.922214, 0.922829, 0.921600 | 0.922214 | -0.02% | 0.13% | no gain |
| Longer batched D=128 / 64×128 | 10 | 0.921293, 0.922010, 0.921088 | 0.921293 | 0.08% | 0.10% | below required 5.88% threshold |

**Decision:** keep `LOAD_LATENCY=0` as the global default. `LOAD_LATENCY=10` is a strong, repeatable candidate for medium D=128 / 64×64, but it will not be surfaced as a recommended setting until a matched Nsight Compute comparison of hint 0 vs 10 is inspected. The longer-batched case stays at 0 because its best timing gain is much smaller than the observed across-session spread.

## Baseline controls and limitations

| Workload | Unfused FP32 CuPy median ms | Baseline P90 ms | Median bracket drift | Maximum bracket drift |
|---|---:|---:|---:|---:|
| short_d32 | 0.297270 | 0.307270 | 1.79% | 77.46% |
| short_d64 | 0.296243 | 0.313247 | 0.28% | 1.69% |
| short_multihead_d64 | 0.280320 | 0.291347 | 4.48% | 4.93% |
| medium_d64 | 0.269562 | 0.279173 | 0.15% | 1.32% |
| medium_d128 | 0.390861 | 0.396831 | 10.47% | 10.83% |
| long_d128 | 4.390656 | 4.397619 | 0.02% | 0.03% |
| longer_batched_d128 | 17.104077 | 17.115771 | 0.03% | 0.08% |

Limits that matter to the result:

- The median/P90 and speedups exclude the first-call wall time; the benchmark script may record a compiler-cache hit as “JIT/first-call”. Long and longer-batched entries have null first-call values in the raw records because those specializations had already been seen in those processes. Short-D=32 and some medium-D=128 tile first-call values show cache/JIT variability. A deliberate clean-cache JIT study is still open.
- Some shorter/medium workloads have large timing or baseline-bracket spread; those are not used as headline results.
- This project baseline is intentionally unfused FP32 CuPy, not an optimized attention vendor reference.
- PyTorch integration and SDPA comparison are separate, unfinished gates. On the pinned torch 2.14.0+cu130 wheel, a plain float32 torch.bmm fails with CUBLAS_STATUS_INVALID_VALUE before the cuTile bridge is invoked. No PyTorch speedup, backend identity or successful-gradient claim follows from this artifact.
- The current repeat native workflow later stopped in the optional GQA/MQA/packed-sequence step without a retrievable final log/artifact. It is not counted as a regression pass or used to overwrite the successful evidence above.

## Reproduction and linked artifacts

- [Native run #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438)
- [Raw artifact #12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195)
- [Dedicated profiler artifact #12261445684](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490/artifacts/12261445684): eight NCU reports, Details/Raw CSV, seven genuine GUI captures; matched hint-10 report is still queued.
- [Current project two-pager](PROJECT_TWO_PAGER.md)
- [Execution evidence index](EXECUTION_EVIDENCE_INDEX.md)
- [Full technical/experiment report](../Report.md)

All figures are from saved artifacts generated by the cited workflow; there is no reliance on manually entered or estimated execution output.