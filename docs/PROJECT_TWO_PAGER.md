# Fused causal attention with NVIDIA cuTile

## Problem

Standard attention forms an N-by-N score matrix, applies a causal mask and softmax, then multiplies the probabilities by V. Writing the full score and probability matrices to GPU memory adds traffic and storage.

## Kernel

The project implements the forward pass of causal scaled dot-product attention in one cuTile kernel. It loads a query tile, streams through key/value tiles, masks future positions and keeps a running max, softmax sum and weighted-value accumulator. That avoids storing the full score matrix.

Q/K/V and the output use FP16. Score accumulation and the online-softmax state use FP32. The required head dimensions are 32, 64 and 128. The current interface requires supported tile-divisible sequence lengths.

## Source map

| File | Purpose |
|---|---|
| src/cutile_attention.py | cuTile kernel and launch path |
| src/reference.py | FP32 NumPy reference |
| src/online_reference.py | Tiled online-softmax reference |
| src/baseline.py | Unfused FP32 CuPy baseline |
| tests/ | Correctness and input-contract tests |
| run_attention.py | CLI |
| benchmarks/ | Timing and profiler harnesses |

The ordinary CUDA implementations in the repository are separate controls. They are not cuTile runs.

## Evidence so far

The last complete RTX 4060 run (#39422402438, commit bf5a623) passed the 48 core GPU tests, 29 numerical configurations and CLI verification. It also ran three benchmark sessions and captured two Nsight Compute configurations.

The kernel was compacted after that run, so current-main correctness still needs a fresh native run. Existing timing and profile results should be described as historical until that run passes.

The performance baseline is deliberately unfused FP32 CuPy. No speedup against FlashAttention or optimized PyTorch SDPA is claimed.

## Remaining work

1. Revalidate current main on the RTX 4060.
2. Measure clean-cache first-call/JIT time for long specializations.
3. Complete the framework attention comparison.
4. Save detailed Nsight Compute GUI views and compare matched tile choices.
5. Make one profile-driven change and test correctness, timing and profiling before keeping it.
6. Update the report and record the real team contributions.

For commands and results, see the [README](../README.md), [course checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md), [test matrix](TEST_MATRIX.md) and [full report](../Report.md).
