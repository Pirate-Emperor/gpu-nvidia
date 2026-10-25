# Fused Causal Self-Attention Using NVIDIA cuTile

**IIT Madras GPU Programming Course Project**  
**Report date:** 22 October 2026  
**Validated platform:** NVIDIA GeForce RTX 4060, native Windows x64, compute capability 8.9  
**Implementation:** Python, NVIDIA cuTile, CuPy and NumPy

## Abstract

This project implements causal scaled dot-product self-attention as a fused NVIDIA cuTile GPU kernel. The kernel processes query, key and value tensors in tiles and uses an online-softmax recurrence. It avoids writing the complete sequence-by-sequence score and probability matrices to global memory.

The recorded native RTX 4060 validation passed all 48 required GPU test executions and all 29 supported workload/tile numerical comparisons. The maximum absolute output error against an independent FP32 NumPy reference was 0.0009765625, with no tolerance violations. Three benchmark sessions compared five tile shapes across seven workloads. The measurements support 64×64 as the general default and 64×128 as a workload-specific candidate for longer batched inputs. A separate load-latency experiment identified a timing candidate for medium D=128 input. That setting has not been promoted to the default pending a matched profiler comparison.

## 1. Objective and scope

The objective was to implement and validate the forward pass of causal self-attention using NVIDIA cuTile. Attention computes

\[
O = \operatorname{softmax}\left(\frac{QK^T}{\sqrt D}+M\right)V
\]

where the causal mask M assigns negative infinity to keys positioned after the corresponding query.

The validated core accepts contiguous FP16 Q, K and V tensors, supports head dimensions D in {32, 64, 128}, requires Dv = D, and returns FP16 output. The sequence length must be divisible by both selected tile dimensions. Partial edge tiles and backward propagation are outside the validated core contract.

Validation was performed on the native Windows RTX 4060, with driver 616.92, Python 3.12.10, CuPy 14.2.0 and NVIDIA cuTile Python 1.6.0. The Linux A100 system with driver R550 was retained for ordinary CUDA C++ control experiments; it is not used as evidence of actual cuTile execution.

## 2. Kernel implementation

### 2.1 Fused tile processing

One cuTile program instance owns a query tile for one batch item, one attention head and one query-block index. It loads a query block and streams the relevant key/value blocks. For each block, it computes scaled query-key scores, applies the causal mask, updates the softmax state, accumulates weighted value vectors, and writes the output block.

A conventional implementation materialises score and probability matrices containing N×N entries per head. The fused kernel instead holds temporary scores for one tile and does not write the complete matrices to global memory.

### 2.2 Online softmax and precision

For each query row, the kernel maintains a running maximum m, a normalisation sum ell, and a weighted-value accumulator u. For a new score tile S_t, it computes a new maximum m' = max(m, max(S_t)). The previous state is rescaled before the current tile is combined:

\[
\alpha=e^{m-m'},\qquad P_t=e^{S_t-m'}
\]

\[
\ell'=\alpha\ell+\sum_j P_{t,j},\qquad
u'=\alpha u+P_tV_t.
\]

Causally masked entries contribute zero. At the end of the key/value loop, the output block is u/ell. Rescaling keeps the contributions from different tiles on a common exponential scale without retaining every score.

Q/K/V and final output use FP16 storage. Score accumulation, running-softmax state, normalisation and the probability-value matrix accumulation use FP32. Probability operands are converted to FP16 for the second matrix multiplication while its accumulator remains FP32. This choice was retained after numerical validation.

The kernel also skips key/value tiles that lie entirely beyond the causal boundary for a query block. Input dimensions, data layout, supported head dimensions and tile divisibility are checked before launch.

### 2.3 Comparison implementation

Performance is compared with a deliberately unfused FP32 CuPy implementation. It materialises the score and probability matrices, applies the causal mask, computes a stable softmax, and multiplies the probabilities by V. Inputs remain resident on the GPU during timed calls.

This is a structural performance baseline, not a vendor-optimised attention implementation. Reported speedups are relative to this baseline only; they are not claims of superiority over FlashAttention or optimised PyTorch scaled dot-product attention.

### 2.4 Independent numerical reference

Correctness is checked against a separate full-matrix FP32 NumPy implementation. The expected output is produced independently of the cuTile result. The acceptance rule is

\[
|O_{\mathrm{actual}}-O_{\mathrm{reference}}|
\le 0.02+0.02|O_{\mathrm{reference}}|.
\]

## 3. Correctness validation

The required GPU suite comprises 48 parameterised test executions from tests/test_gpu_correctness.py and tests/test_gpu_edge_cases.py. The workflow sets REQUIRE_CUTILE=1, so missing GPU dependencies cannot turn the required tests into silent skips.

| Test group | Executions | Property checked |
|---|---:|---|
| Mandatory workload/tile comparisons against independent FP32 NumPy | 29 | Numerical agreement for each supported case/tile pair |
| Additional asymmetric tile | 1 | Numerical agreement for an additional supported tile |
| Small- and large-score inputs | 2 | Numerical stability |
| Future-value invariance in the correctness suite | 1 | Future values cannot affect earlier queries |
| Non-divisible sequence rejection in the correctness suite | 1 | Enforces fixed-tile requirements |
| Constant V for D=32, 64 and 128 | 3 | Weighted averaging preserves a constant V tensor |
| Zero-score prefix average | 1 | Equal scores yield the average of the causal prefix |
| Tile-to-tile agreement | 1 | Different supported tiles agree with the reference |
| Future-value and future-key invariance in edge tests | 2 | Causal isolation |
| NaN and infinity rejection | 3 | Invalid values are rejected before computation |
| Non-contiguous input rejection | 1 | Enforces memory-layout requirements |
| Unsupported or mismatched dimensions | 2 | Enforces the shape contract |
| Additional non-divisible sequence checks | 1 | Rejects invalid query/key tile combinations |
| **Total** | **48** | **All required executions passed** |

The separate numerical matrix checked every supported case/tile pair against the independent FP32 NumPy reference.

| Numerical metric | Recorded result |
|---|---:|
| Supported case/tile configurations passed | 29 / 29 |
| Tolerance violations | 0 |
| Maximum absolute output error | 0.0009765625 |
| Largest per-configuration 99th-percentile absolute error | 0.00048828125 |
| Largest per-configuration RMSE | 0.000086860741 |
| Maximum relative error for reference magnitude ≥ 0.01 | 2.2459% |
| Maximum error as a percentage of its permitted tolerance | 2.4367% |

The last value is the percentage of the permitted tolerance used by the worst error. It is not the percentage error of the output.

The command-line interface was checked with deterministic FP16 input of shape (1,1,64,32) and seed 20261018. The saved output had the expected shape and datatype, contained only finite values, and passed the independent NumPy comparison.

## 4. Performance methodology

Seven fixed-seed workloads cover short, medium, long and batched-long sequences with dimensions 32, 64 and 128. Five tile candidates were tested wherever supported: 32×64, 64×64, 64×128, 128×64 and 128×128. This produced 29 supported case/tile pairs.

Each case/tile pair was measured in three separate benchmark sessions. Every session used 10 warm-up calls, 30 CUDA-event timing samples, and five kernel launches per sample. Each sample's elapsed time was divided by five. There were 150 timed launches per case/tile pair in each session, in addition to the warm-ups.

The tile order was changed between sessions:

| Session | Tile order |
|---|---|
| 1 | 32×64, 64×64, 64×128, 128×64, 128×128 |
| 2 | 64×64, 64×128, 128×64, 128×128, 32×64 |
| 3 | 128×128, 128×64, 64×128, 64×64, 32×64 |

Changing the order reduces the risk that a tile always measured first or last benefits from a consistent ordering effect. It does not eliminate all system variation.

A CUDA event is a timing marker recorded in the GPU command stream. Thirty samples provide a distribution rather than a single timing. The median is the middle sample after sorting the values. P90 is the 90th percentile, below which approximately 90% of samples fall.

The unfused baseline was measured before and after each tile sweep. Baseline variation is the absolute difference between the two baseline medians divided by their average, expressed as a percentage. A large value indicates that performance conditions changed during the session and makes small differences between tiles less decisive.

First-call wall time is reported separately from steady-state latency. It can include compilation and runtime setup, but can also reuse previously compiled code. It is therefore not presented as a controlled clean-cache compilation time.

## 5. Three-session tile results

The table reports the recorded aggregate medians from the completed native RTX 4060 benchmark. Latencies are in milliseconds. Speedup is relative to the unfused FP32 CuPy baseline. Session range is the difference between the highest and lowest session medians divided by the aggregate median. Maximum baseline variation is the largest before/after difference observed during a tile sweep.

| Workload | Shape (B,H,N,D) | Tile | Median (ms) | P90 (ms) | Baseline (ms) | Speedup | Session range | Max baseline variation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short D=32 | (1,1,64,32) | 32×64 | 0.015770 | 0.018473 | 0.297270 | 19.10× | 2.09% | 77.46% |
| Short D=32 | (1,1,64,32) | 64×64 | 0.015974 | 0.018125 | 0.297270 | 18.73× | 0.64% | 77.46% |
| Short D=64 | (1,1,64,64) | 32×64 | 0.016074 | 0.018186 | 0.296243 | 18.25× | 5.20% | 1.69% |
| Short D=64 | (1,1,64,64) | 64×64 | 0.016282 | 0.018862 | 0.296243 | 18.33× | 1.36% | 1.69% |
| Short multi-head D=64 | (1,4,128,64) | 32×64 | 0.016074 | 0.018810 | 0.280320 | 16.96× | 3.70% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 64×64 | 0.016179 | 0.018452 | 0.280320 | 17.15× | 1.29% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 64×128 | 0.016566 | 0.019313 | 0.280320 | 16.96× | 3.71% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 128×64 | 0.016794 | 0.018043 | 0.280320 | 16.69× | 3.66% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 128×128 | 0.018842 | 0.020910 | 0.280320 | 14.88× | 4.35% | 4.93% |
| Medium D=64 | (1,8,256,64) | 32×64 | 0.020070 | 0.021238 | 0.269562 | 14.00× | 95.92% | 1.32% |
| Medium D=64 | (1,8,256,64) | 64×64 | 0.017171 | 0.017833 | 0.269562 | 16.25× | 109.73% | 1.32% |
| Medium D=64 | (1,8,256,64) | 64×128 | 0.019456 | 0.020480 | 0.269562 | 14.40× | 90.05% | 1.32% |
| Medium D=64 | (1,8,256,64) | 128×64 | 0.019430 | 0.022733 | 0.269562 | 14.22× | 120.16% | 1.32% |
| Medium D=64 | (1,8,256,64) | 128×128 | 0.024944 | 0.026010 | 0.269562 | 11.15× | 76.36% | 1.32% |
| Medium D=128 | (1,8,512,128) | 32×64 | 0.069632 | 0.070659 | 0.390861 | 5.62× | 5.00% | 10.83% |
| Medium D=128 | (1,8,512,128) | 64×64 | 0.058368 | 0.058814 | 0.390861 | 6.70× | 9.82% | 10.83% |
| Medium D=128 | (1,8,512,128) | 64×128 | 0.058368 | 0.059044 | 0.390861 | 6.70× | 2.46% | 10.83% |
| Medium D=128 | (1,8,512,128) | 128×64 | 0.079053 | 0.079258 | 0.390861 | 4.94× | 5.31% | 10.83% |
| Medium D=128 | (1,8,512,128) | 128×128 | 0.107315 | 0.107950 | 0.390861 | 3.64× | 3.05% | 10.83% |
| Long D=128 | (1,16,1024,128) | 32×64 | 0.334950 | 0.337736 | 4.390656 | 13.11× | 0.06% | 0.03% |
| Long D=128 | (1,16,1024,128) | 64×64 | 0.267059 | 0.269128 | 4.390656 | 16.49× | 0.92% | 0.03% |
| Long D=128 | (1,16,1024,128) | 64×128 | 0.270848 | 0.273449 | 4.390656 | 16.19× | 1.13% | 0.03% |
| Long D=128 | (1,16,1024,128) | 128×64 | 0.316109 | 0.319324 | 4.390656 | 13.89× | 0.78% | 0.03% |
| Long D=128 | (1,16,1024,128) | 128×128 | 0.411853 | 0.415846 | 4.390656 | 10.66× | 0.79% | 0.03% |
| Longer batched D=128 | (2,8,2048,128) | 32×64 | 1.142989 | 1.146900 | 17.104077 | 14.97× | 0.31% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 64×64 | 0.932966 | 0.936468 | 17.104077 | 18.33× | 0.33% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 64×128 | 0.915558 | 0.921457 | 17.104077 | 18.69× | 0.50% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 128×64 | 1.017139 | 1.020314 | 17.104077 | 16.83× | 0.23% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 128×128 | 1.285120 | 1.289237 | 17.104077 | 13.32× | 0.32% | 0.08% |

The medium D=128 case has shape (1,8,512,128). The longer-batched D=128 case has shape (2,8,2048,128). Both use D=128 and eight attention heads, but the longer-batched case doubles batch size and quadruples sequence length. It processes eight times as many query positions and approximately 32 times as many causal query-key pairs.

The measurements support 64×64 as the general default. On the long D=128 workload, it recorded 0.267059 ms and 16.49× speedup. For longer batched input, 64×128 recorded 0.915558 ms and was about 1.86% faster than 64×64. That improvement is specific to the tested workload. The 64×64 and 64×128 medians tie for medium D=128, where maximum baseline variation reached 10.83%. The 128×128 tile was slower than 64×64 on both long D=128 workloads. Short and medium D=64 timings were more variable and should not be used as precise headline claims.

## 6. Load-latency hint experiment

A second experiment varied the cuTile load-latency hint while keeping workload and tile shape fixed. Values 0, 2, 4, 6, 8 and 10 were tested for medium D=128 with tile 64×64 and longer-batched D=128 with tile 64×128. Every variant passed the independent numerical check before timing. The experiment used three sessions, 10 warm-ups, 30 timing samples and five launches per sample, for 36 hint/session trials.

| Workload | Hint | Session medians (ms) | Aggregate median (ms) | Improvement vs hint 0 | Decision |
|---|---:|---|---:|---:|---|
| Medium D=128, 64×64 | 0 | 0.058061, 0.058342, 0.058368 | 0.058342 | Reference | Keep default |
| Medium D=128, 64×64 | 2 | 0.055296, 0.054886, 0.054682 | 0.054886 | 5.92% | Candidate; below best recorded result |
| Medium D=128, 64×64 | 4 | 0.058368, 0.058368, 0.058368 | 0.058368 | -0.04% | No improvement |
| Medium D=128, 64×64 | 6 | 0.058368, 0.058448, 0.058368 | 0.058368 | -0.04% | No improvement |
| Medium D=128, 64×64 | 8 | 0.056525, 0.056397, 0.056515 | 0.056515 | 3.13% | Candidate; below best recorded result |
| Medium D=128, 64×64 | 10 | 0.053325, 0.053248, 0.053043 | 0.053248 | 8.73% | Timing candidate; matched profiler review pending |
| Longer batched D=128, 64×128 | 0 | 0.992256, 0.920883, 0.922010 | 0.922010 | Reference | Keep default |
| Longer batched D=128, 64×128 | 2 | 0.957542, 0.920576, 0.920576 | 0.920576 | 0.16% | Below measurement threshold |
| Longer batched D=128, 64×128 | 4 | 0.922214, 0.920883, 0.920576 | 0.920883 | 0.12% | Below measurement threshold |
| Longer batched D=128, 64×128 | 6 | 0.921498, 0.920576, 0.920678 | 0.920678 | 0.14% | Below measurement threshold |
| Longer batched D=128, 64×128 | 8 | 0.922214, 0.922829, 0.921600 | 0.922214 | -0.02% | No improvement |
| Longer batched D=128, 64×128 | 10 | 0.921293, 0.922010, 0.921088 | 0.921293 | 0.08% | Below measurement threshold |

A candidate is accepted only when its median improvement exceeds both a 1% minimum and half the combined default/candidate session range. Hint 10 for medium D=128 improved median latency by 8.73% with a 0.53% range. The matched profiler comparison is still required before promoting the hint. On the longer-batched workload, the largest observed gain was below the measurement threshold. The global default therefore remains LOAD_LATENCY=0.

## 7. Profiler measurements

Two actual cuTile configurations were measured with NVIDIA Nsight Compute. Profiler collection adds overhead, so these durations are not used as ordinary benchmark medians.

| Workload and tile | Profiled duration | Registers per thread | Static shared memory per block | Achieved occupancy | SM throughput vs peak | DRAM throughput vs peak | Scheduler cycles with no eligible warp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medium D=128, 64×64 | 52.90 microseconds | 231 | 65.54 KiB | 8.35% | 18.81% | 22.88% | 85.83% |
| Longer batched D=128, 64×128 | 984.26 microseconds | 255 | 81.92 KiB | 8.33% | 30.27% | 11.89% | 85.26% |

Shared-memory use substantially limits the number of blocks that can be active on a multiprocessor; measured occupancy is approximately 8.3% in both configurations. The high percentage of scheduler cycles without an eligible warp indicates frequent instruction-issue gaps. These measurements motivate further work on resource use and load scheduling but do not identify a unique source-level cause. Nsight's estimated local-speedup values are rule-based estimates, not measured gains. The matched profile for hint 10 remains pending, so the hint is not promoted.

## 8. Additional experiments and limitations

The repository retains an earlier ordinary CUDA C++ implementation ladder run on the A100 control system. Those results are not cuTile measurements and are not combined with the RTX 4060 performance table. The v4 persistent-query-fragment variant passed its correctness suite but showed mixed timing and no clear resource advantage. The v5 half2-oriented load variant passed its correctness suite, but long-workload improvements were only about 0.3% to 0.6%; short-workload results were mixed. Neither variant was promoted.

The PyTorch scaled dot-product attention comparison remains incomplete. The installed PyTorch CUDA 13.0 build failed in a plain torch.bmm call with CUBLAS_STATUS_INVALID_VALUE before the cuTile integration path was reached. No framework timing, backend identification or speedup is claimed.

A controlled clean-cache compilation study remains incomplete for the longest specialisations. First-call timings are reported separately and are not presented as exact compilation times.

## 9. Conclusion

The project provides a validated fused cuTile forward implementation for causal self-attention on the native RTX 4060. All 48 required GPU test executions passed, as did all 29 supported case/tile numerical comparisons. No numerical tolerance violations were observed.

The strongest recorded long-workload result was 0.267059 ms at tile 64×64, equivalent to 16.49× the performance of the deliberately unfused FP32 CuPy baseline. For longer batched D=128 input, tile 64×128 recorded 0.915558 ms and 18.69× baseline speedup. The 64×64 tile remains the general default. The 64×128 tile is a measured candidate for the longer batched case. The medium-workload load-latency hint remains experimental pending matched profiler review.

## References

1. NVIDIA, cuTile Python Documentation. https://docs.nvidia.com/cuda/cutile-python/
2. NVIDIA, cuTile Python Performance Guide. https://docs.nvidia.com/cuda/cutile-python/performance.html
3. NVIDIA, CUDA Compatibility Guide. https://docs.nvidia.com/deploy/cuda-compatibility/index.html
4. Project source and recorded results. https://github.com/xemptynothing/gpu-nvidia
