# Fused Causal Attention with NVIDIA cuTile — Technical Report

**Evidence snapshot:** 19 October 2026  
**Implementation:** src/cutile_attention.py  
**Target:** NVIDIA GeForce RTX 4060, native Windows x64, compute capability 8.9  
**Core validation:** [GitHub Actions run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223), commit cd1ec63b210682613ddb29dca3543f0cacc9bacf  
**Profiler export:** [run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066), [artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039)

> **Status:** The required native cuTile correctness, numerical, CLI, controlled benchmark and two-configuration profiling gates have evidence. The course assignment is not fully closed yet: the framework/vendor attention reference has not been measured; cold/JIT timings are missing for already-compiled long cases; Week 3’s second profiler-guided optimization is not yet demonstrated; and the individual-contribution statement needs author confirmation. This report deliberately does not call those items complete.

## 1. Executive summary

This project implements causal scaled dot-product self-attention as a genuine NVIDIA cuTile Python fused GPU kernel. The kernel keeps the full sequence-by-sequence score and probability matrices out of global memory, updates an online-softmax state across K/V tiles, skips wholly future tiles, and writes FP16 outputs.

Native RTX 4060 validation passed: a real cuTile smoke kernel checked 4,096 outputs exactly; the required attention suite passed 48/48 tests with REQUIRE_CUTILE=1; the independent numerical report passed all 29 legal case/tile configurations; the CLI passed an independent NumPy oracle; three controlled benchmark sessions completed; and four compressed Nsight Compute profiles were collected and inspected. There was no RTX 4060 ERR_NVGPUCTRPERM.

The strongest repeatable result is the longer batched D=128 case at tile 64×128: **0.919347 ms median, 0.924795 ms P90, 18.62× versus the deliberately unfused FP32 CuPy baseline**, with 0.7% three-session range and 0.08% maximum bracketed baseline drift. This is not a comparison with FlashAttention or vendor-optimized SDPA.

## 2. Objective and kernel design

Causal self-attention is

\[
s_{ij}=\begin{cases}\frac{q_i\cdot k_j}{\sqrt{D}},&j\le i\\-\infty,&j>i\end{cases},\quad
p_{ij}=\frac{e^{s_{ij}}}{\sum_{j'\le i}e^{s_{ij'}}},\quad
o_i=\sum_{j\le i}p_{ij}v_j.
\]

Q/K/V use FP16 storage and logical layout [B,H,N,D]. QKᵀ accumulations, running-softmax state, normalizers and P×V accumulator are FP32; probabilities are converted to FP16 operands for P×V; outputs are FP16. Supported dimensions are D ∈ {32,64,128}. Sequence lengths must divide the selected tile dimensions; partial edge tiles are outside the fixed assignment scope.

### Online-softmax derivation

For a query row, maintain a running maximum (m), normalization sum (ell), and unnormalized weighted-value accumulator (u). Initially (m=-\infty), (ell=0), and (u=0). For the next masked score tile (S_t), calculate its maximum (m_t), then (m'=\max(m,m_t)). Rescale the prior state by (\alpha=e^{m-m'}) and let (P_t=e^{S_t-m'}), with causally masked entries set to zero. Update

\[
\ell'=\alpha\ell+\sum_j P_{t,j},\qquad u'=\alpha u+P_tV_t.
\]

Set (m\leftarrow m'), (\ell\leftarrow\ell'), (u\leftarrow u'); after the last K/V tile, output (o=u/\ell). Rescaling the old numerator and denominator to the new maximum lets each tile join the same normalized softmax result without storing an N×N matrix. The running maximum stabilizes the exponentials.

A program instance owns one query tile for a batch/head and query-block index. It loads Q[Tq,D], iterates K[Tk,D] and V[Tk,D], creates a temporary score tile [Tq,Tk], applies the causal mask, updates per-row softmax state, and stores O[Tq,D]. The loop bound avoids K/V tiles wholly beyond the final causal position in the query tile.

## 3. Correctness and numerical accuracy

The required workflow explicitly sets REQUIRE_CUTILE=1, so a missing cuTile import is a hard failure, not a silent GPU skip. A full-matrix FP32 NumPy oracle independently checks the output under the fixed mixed tolerance

\[
|O_{\mathrm{actual}}-O_{\mathrm{reference}}|\le 0.02+0.02|O_{\mathrm{reference}}|.
\]

| Validation item | Evidence |
|---|---:|
| Genuine cuTile smoke | PASS; exact check over 4,096 outputs |
| Actual-cuTile attention suite | 48 passed, 0 failed |
| Legal case/tile accuracy configurations | 29 passed, 0 tolerance violations |
| Maximum absolute output error | 0.0009765625 |
| Largest per-configuration P99 absolute error | 0.00048828125 |
| Largest per-configuration RMSE | 0.000086860741 |
| Maximum relative error where abs(reference) ≥ 0.01 | 0.02245863 (2.25%) |
| Maximum error / permitted tolerance | 0.024366 (2.44% of tolerance budget) |
| CLI output | FP16, finite, shape-valid; NumPy oracle PASS |

The 2.44% number is the fraction of the permitted tolerance envelope consumed by the worst element; it is not 2.44% output error. The CLI generated deterministic FP16 Q/K/V of shape [1,1,64,32], seed 20261018, and verified shape, dtype, finiteness and oracle agreement. Test and JSON/log evidence is preserved in [core artifact #12149980979](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979).

## 4. Performance methodology and results

Seven fixed workloads and up to five legal tile candidates were tested: 32×64, 64×64, 64×128, 128×64 and 128×128. Each of three independent sessions used ten warmups and 30 CUDA-event samples per configuration. Each timed sample averaged five launches, for 150 timed launches per case/tile/session. Tile order was rotated/reversed between sessions. The deliberately unfused FP32 CuPy baseline was measured before and after the sweep to expose timing drift.

| Workload | Tile | Fused median (ms) | Fused P90 (ms) | Unfused baseline (ms) | Speedup | Three-session range | Max baseline drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short D=32 | 32×64 | 0.015750 | 0.017747 | 0.308131 | 19.56× | 197.0% | 81.17% |
| Short D=64 | 64×64 | 0.016282 | 0.018227 | 0.308677 | 18.71× | 196.2% | 38.80% |
| Short multi-head D=64 | 64×128 | 0.015456 | 0.019041 | 0.288666 | 18.68× | 20.7% | 2.27% |
| Medium D=64 | 64×64 | 0.017715 | 0.021709 | 0.288814 | 16.12× | 6.9% | 4.58% |
| Medium D=128 | 64×128 | 0.056934 | 0.058959 | 0.374733 | 6.58× | 10.0% | 6.09% |
| Long D=128 | 64×128 | 0.273203 | 0.277914 | 4.403866 | 16.15× | 3.1% | 0.22% |
| Long batched D=128 | 64×128 | 0.919347 | 0.924795 | 17.126861 | 18.62× | 0.7% | 0.08% |

The general default remains 64×64. For long D=128, the 64×64 median (0.273408 ms) and 64×128 median (0.273203 ms) differ by only about 0.08%, too little to justify a global default switch. For long-batched D=128, 64×128 is around 1.8% faster than 64×64 (0.919347 vs 0.936448 ms), with less than 1% session range; it is a promising specialization, but should remain workload-specific until independently reconfirmed.

Short-case numbers are not reliable headline results: Windows/WDDM scheduling and launch overhead produce up to 197% fused range and 81.17% baseline drift. Output throughput in the benchmark JSON is based on output bytes; do not describe it as total memory bandwidth.

### Cold/JIT timing caveat

The harness records first-call wall time separately when available, but these values may include compiler cache hits and are JIT-associated latency, not pure compile time. Medium D=128 first calls were roughly 27–32 ms. For long D=128 and long-batched D=128, the session-level JIT fields are null because those specializations were already compiled. A deliberately controlled clean-cache measurement, or an approved explanation for omitting it, remains open.

## 5. Nsight Compute profiling and interpretation

Nsight Compute CLI 2026.3.1.0 successfully captured four reports: basic throughput/resource profiles and supplemental SchedulerStats/WarpStateStats profiles for medium_d128, tile 64×64, and longer_batched_d128, tile 64×128. The profile artifact includes compressed .ncu-repz files, Details exports, Raw CSV files and status/command logs. The downloaded 289,372-byte ZIP matched GitHub’s SHA-256 digest fd947285bc52731a97127fda08a86a9e53dc9a8000c9164966165fed0fc06b1c.

| Metric | medium D=128, 64×64 | long batched D=128, 64×128 |
|---|---:|---:|
| Tensor shape | B=1,H=8,N=512,D=128 | B=2,H=8,N=2048,D=128 |
| NCU kernel duration | 52.90 µs | 984.26 µs |
| Grid / threads per block | 64 CTAs / 128 | 512 CTAs / 128 |
| Registers per thread | 231 | 255 |
| Static shared memory per CTA | 65.54 KiB | 81.92 KiB |
| Allocated shared memory per CTA | ~66.56 KiB | ~82.94 KiB |
| Theoretical / achieved occupancy | 8.33% / 8.35% | 8.33% / 8.33% |
| SM compute throughput | 18.81% peak | 30.27% peak |
| DRAM throughput | 22.88% peak | 11.89% peak |
| L1/TEX / L2 throughput | 17.35% / 14.52% | 21.00% / 20.07% |
| Scheduler cycles with no eligible warp | 85.83% | 85.26% |
| Eligible / active warps per scheduler | 0.14 / 1.00 | 0.15 / 1.00 |
| Warp cycles per issued instruction | 7.08 | 6.78 |
| NCU rule pipeline-stall estimate | 2.4 cycles | 3.1 cycles |
| NCU rule estimated local speedup | 33.57% heuristic | 46.04% heuristic |

**Interpretation:** shared memory is the clearest CTA-residency constraint. The occupancy section attributes one CTA per SM and roughly 8.3% occupancy to the kernels’ shared-memory footprint. Register usage is also high. Neither workload approaches peak compute or DRAM throughput, and only 14–15% of scheduler cycles have one or more eligible warps. Evidence therefore motivates hypothesis-driven experiments on shared-memory/register footprint and instruction scheduling.

The NCU “estimated local speedup” values are rule heuristics, not measured gains or guarantees. The optional smsp__pcsamp_sample_count metric was unavailable, so detailed PC/source-level stall attribution is not supported. The automated evidence contains CSV/Details exports and derived plots, not a literal Nsight GUI screenshot. Native .ncu-repz reports are preserved for interactive inspection in Nsight Compute.

## 6. Optimization history and engineering decisions

Two optimizations were retained only after correctness checks passed:

1. **Causal-loop pruning:** skip K/V tiles wholly beyond the last causal position instead of loading them and masking all their scores.
2. **Mixed-precision P×V operands:** probabilities are rounded to FP16 operands for the second matrix multiply while QK accumulation, online-softmax state, normalizers and P×V accumulator remain FP32.

No kernel-code change was accepted solely from the newly captured profiles. The report supports an experiment proposal, not a claim of a third proven optimization. The Week 3 milestone should test a specific shared-memory/register/scheduling hypothesis, preserve an exact code diff, run targeted and full correctness first, then alternating-order benchmark repeats and comparable before/after NCU profiles. Promote only if gain exceeds measurement noise without regressions.

## 7. Assignment closure, limitations and next actions

| Rubric item | Current status | Required next action |
|---|---|---|
| Fused causal cuTile kernel, online softmax, no global N×N matrix | PASS | Keep as baseline; do not alter without evidence |
| D=32/64/128, FP16 input/output, FP32 accumulation | PASS in tested matrix | Preserve validation gates |
| Independent correctness and error report | PASS: 48 tests + 29 configs | Keep raw logs/JSON |
| CLI and NumPy oracle | PASS | Keep deterministic oracle check |
| Unfused baseline, multiple tiles, 3 controlled sessions | PASS with caveats | Preserve JSON/CSV and noise caveats |
| Two real cuTile workload profiles | PASS | Keep compressed reports and exports |
| Cold/JIT timings for each workload | PARTIAL | Clean-cache measurement for long cases or documented exception |
| Framework/vendor attention performance reference | NOT RUN | With owner approval, compare to a CUDA-enabled PyTorch SDPA reference and record the actual backend/kernel; the PDF says this is not part of the speedup grade |
| Week 2 profiler-guided optimization | PARTIAL | Trace an optimization choice explicitly to profile evidence |
| Week 3 second profiler-guided improvement | OPEN | Correctness-first experiment with before/after benchmarking and profiling |
| Final report 5–8 pages | 7-page PDF prepared | Confirm required author/team metadata |
| Individual contributions | NEEDS OWNER INPUT | Insert actual names and a truthful contribution split |

### Remaining and optional objectives

The assignment’s possible next step is to integrate the fused kernel into an example PyTorch model and measure output agreement and end-to-end improvement over that model’s baseline. Other optional extensions identified in the project materials are BF16/FP8, backward propagation, variable-length/packed sequences, GQA/MQA, autotuning, CuTe DSL comparison (instructor-approved alternative track), and architecture-specific pipelining/async data movement. These should not displace the remaining rubric gaps. Do not install PyTorch or other packages without explicit approval.

### Reproducibility links

- [Native RTX 4060 validation run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223)
- [Core run artifact #12149980979](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979)
- [Dedicated Nsight Compute run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066)
- [Four-profile NCU artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039)
- [Detailed numerical/benchmark snapshot](CUTILE_RTX4060_RESULTS_2026-10-19.md)
- [Profiler analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md)
- [Rubric-mapped checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md)
- Original problem description: nvidia2-cutile.pdf in repository root.

**Individual contributions:** [Owner/team to confirm. No authorship attribution is guessed in this report.]
