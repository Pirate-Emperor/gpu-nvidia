# Course project checklist

Source: nvidia2-cutile.pdf, the six-page course project brief.  
Updated: 24 October 2026.

This checklist covers the required implementation and evaluation work. The optional extension list is deliberately left out.

## 1. Kernel and interface

- [x] Implement forward causal scaled dot-product attention in NVIDIA cuTile.
- [x] Use FP16 Q/K/V and output; keep score and online-softmax accumulation in FP32.
- [x] Support D = 32, 64 and 128 for legal tile-divisible shapes.
- [x] Fuse score calculation, scaling, causal mask, softmax and value accumulation.
- [x] Avoid storing the full N-by-N score or probability matrix in global memory.
- [x] Provide a CLI that reads an NPZ input and writes an NPY output.

## 2. Correctness

- [x] Compare outputs with an independent FP32 NumPy reference.
- [x] Cover all required dimensions and several query/key tile shapes.
- [x] Check causal masking, small/large values, output finiteness and invalid inputs.
- [x] Run the 48-test GPU suite with missing cuTile treated as a failure.
- [x] Check 29 legal workload/tile combinations against the mixed tolerance: abs(error) <= 0.02 + 0.02 * abs(reference).

The last full GPU pass is run #39422402438, on commit bf5a623. The main kernel was compacted afterward; rerun it before treating that evidence as current.

## 3. Performance

- [x] Keep an unfused FP32 CuPy baseline.
- [x] Benchmark several legal tile shapes on short, medium and long workloads.
- [x] Use warmups, CUDA events, at least 30 samples, repeated sessions and rotated tile order.
- [x] Record medians, tail latency and timing variation.
- [ ] Measure first-call/JIT time with a clean cache for the long specializations.
- [ ] Complete a compatible framework/vendor attention comparison and record the backend that ran.

Reported speedups so far are against the unfused CuPy baseline, not FlashAttention or optimized PyTorch SDPA.

## 4. Nsight Compute and optimization

- [x] Capture two actual-cuTile workload/tile configurations.
- [x] Record registers, shared memory, occupancy, throughput and scheduler/warp-state metrics.
- [ ] Save detailed GUI captures for Summary, Details, Raw, Occupancy, Memory, Scheduler and Warp State views.
- [ ] Pick a change from the profile, rerun correctness, benchmark it in repeated sessions and capture a matching before/after profile.
- [ ] Keep or reject the change based on the measured result.

## 5. Final report

- [x] Keep the source, tests, timing scripts and existing evidence in the repository.
- [ ] Rerun current-source checks and update the report with the latest results.
- [ ] State limitations and negative results clearly.
- [ ] Record each team member's actual contribution.

Related evidence: [test matrix](TEST_MATRIX.md), [profiler analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md), [full report](../Report.md).
