# Fused causal attention with NVIDIA cuTile

## Problem

Conventional attention forms an N×N score matrix, applies a causal mask and softmax, then multiplies by V. Writing the full score/probability matrices adds global-memory traffic and storage.

## Kernel

The project computes causal scaled dot-product attention in a fused NVIDIA cuTile kernel. It loads a query tile, streams through key/value tiles, masks future positions and maintains a running maximum, normalization sum and weighted-value accumulator using online softmax. The implementation skips K/V tiles beyond the causal boundary and does not materialise the full N×N score/probability matrices in global memory.

Q/K/V and the tested output are FP16; score, online-softmax and P×V accumulation are FP32. The PDF allows `D` and `Dv` independently in `{32,64,128}`, and specifies `1≤B≤16`, `1≤H≤32`, `1≤N≤8192`. The host and device validation code now implements those shape rules; CPU tests pass, while current-main native GPU confirmation remains queued.

## Evidence and performance

The saved RTX 4060 timing matrix covers seven workload families and 29 tile/workload pairs across three sessions. The speedups are versus the deliberately unfused FP32 CuPy baseline. Historical examples include 16.49× for long D=128 at 64×64 and 18.69× for longer-batched D=128 at 64×128. These values predate current-source code changes.

No valid PyTorch attention speedups are available. The attempted plain `torch.bmm` probe failed with `CUBLAS_STATUS_INVALID_VALUE` before a framework attention timing. A compatible framework/vendor attention reference remains an assignment performance-evaluation item. The fused kernel contains causal tile pruning and mixed-precision P×V, but current timings do not isolate those techniques in individual A/B ablations.

The A100 `v1`–`v5` comparison is ordinary CUDA C++ on A100 hardware, not cuTile; do not combine those values with the RTX 4060 matrix.

## Nsight Compute

There are 27 native RTX 4060 reports and 27 per-configuration Markdown analyses, plus 972 original GUI screenshots (32 primary and four PTX supplement per profile). A corrected four-pane horizontal-scroll probe passed for one representative report; the full 27-profile proof is incomplete.

All 27 contact sheets were visually checked for coverage and representative full-resolution views were inspected. The requested literal transcription of every instruction and every small visual detail in all 972 full-resolution screenshots is still incomplete. The [manual inspection status](ncu/rtx4060/27-profile-gui-analysis/MANUAL_INSPECTION_STATUS.md) records this limitation.

## Remaining work

1. Complete native RTX 4060 validation on current source, including unequal D/Dv correctness and B/H/N bounds.
2. Produce a compatible, numerically checked framework/vendor attention reference and measure missing clean-cache JIT timings.
3. Complete matched profiler-guided before/after work and the 27-profile horizontal pane proof.
4. Complete the requested full-resolution manual screenshot/instruction transcription and reconcile all per-profile READMEs.
5. Confirm actual contributor names and contributions.

See the [root README](../README.md), [assignment audit](ASSIGNMENT_BRIEF_AUDIT.md), [course completion checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md) and [27-profile NCU analysis](ncu/rtx4060/27-profile-gui-analysis/README.md). The final report has not been created; report drafting should start only after the remaining gates are reconciled and the project owner supplies report-stage input.
