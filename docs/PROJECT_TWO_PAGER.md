# Project Two-Pager — Fused Causal Attention with NVIDIA cuTile

**Updated:** 20 October 2026  
**Target GPU:** NVIDIA GeForce RTX 4060, SM89, native Windows x64  
**Project focus:** A fused, numerically stable causal self-attention forward kernel, plus carefully isolated optional extensions.  
**Product goal:** A reproducible engineering demo that takes Q/K/V tensors, produces causal attention output on a Windows RTX 4060, and provides auditable correctness, latency, and profiler evidence. It is not positioned as a production replacement for FlashAttention or vendor-optimized PyTorch SDPA.

## 1. Problem and solution

Standard scaled dot-product attention forms score and probability matrices of shape (N \times N). Materializing those intermediates creates substantial global-memory traffic. This project implements causal attention as a tiled NVIDIA cuTile Python kernel that fuses (QK^T), scaling, causal masking, online softmax, and the (PV) accumulation. It streams key/value tiles and does not store the full score or probability matrix in global memory.

The attention recurrence maintains running row-wise maximum (m), normalization (l), and weighted-value accumulator (a). For each new score tile it rescales prior state before combining the current tile. This preserves stable softmax without requiring all scores at once. Scores and state accumulate in FP32; supported input tensors are FP16. The core contract is causal forward attention, not a complete production attention library.

## 2. Architecture and source map

| Component | Responsibility |
|---|---|
| `src/cutile_attention.py` | Actual cuTile kernel and public forward entry point; tile loop, mask, online softmax and output accumulation |
| `src/reference.py` | Independent FP32 mathematical oracle |
| `src/online_reference.py` | Independent tiled online-softmax reference |
| `src/extended_attention.py` | Independent references and helpers for optional attention objectives |
| `src/baseline.py` | Deliberately unfused CuPy comparison baseline (not vendor-optimized attention) |
| `run_attention.py` | CLI entry point for generated or supplied inputs |
| `tests/` | Core correctness/edge tests and optional-extension tests |
| `benchmarks/` and `scripts/` | Controlled timing, profiler capture, accuracy reports, framework comparison and autotuning |
| `Report.md` | Detailed technical narrative, benchmark tables, NCU analysis and experiment decisions |
| `docs/EXECUTION_EVIDENCE_INDEX.md` | Workflow/runner diagnostics and links to actual execution evidence |

Keep the native Windows RTX 4060 cuTile target separate from the Linux A100/ordinary-CUDA control environment. A CUDA C++ control kernel is not an actual cuTile execution.

## 3. What has been demonstrated on the core baseline

The recorded native Windows RTX 4060 baseline completed a genuine cuTile smoke test, 48 required GPU correctness tests, an independent numerical matrix of 29 legal case/tile combinations with zero mixed-tolerance violations, CLI output verification against an independent NumPy oracle, three controlled benchmark sessions, and actual Nsight Compute captures for two D=128 workloads. See the [core run](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223), [profiler run](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066), and [full results](CUTILE_RTX4060_RESULTS_2026-10-19.md).

The latest three-session native benchmark (run [#39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), commit `bf5a6233`) measured long D=128 / 64×64 at **0.267059 ms median, 16.49×** versus the deliberately unfused FP32 CuPy baseline (0.9% across-session range), and longer-batched D=128 / 64×128 at **0.915558 ms, 18.69×** (0.5% range). The medium D=128 / 64×64 default measured 0.058368 ms but had a 9.8% range, so its tile sweep is notably noisier. These are not comparisons against FlashAttention or vendor-optimized PyTorch SDPA. Keep **64×64 as the general default** and use 64×128 only for the longer batched workload.

The NCU snapshots show a resource-constrained kernel with low achieved occupancy and many cycles without an eligible warp. Nsight's local speedup/latency estimates are heuristics, not measured speedups. The DRAM-load-latency hint experiment is only promoted if repeated timing, accuracy and profiler evidence support it.

## 4. Optional paths: implemented is not the same as validated

The current source also contains GQA/MQA launch support, a packed-variable-length wrapper, a PyTorch autograd/model bridge, a correctness-gated tile autotuner, and a cuTile DRAM-load-latency hint sweep. These were added after the last fully validated core GPU snapshot.

**Fresh native status — 20 October 2026 (Windows RTX 4060):**

- The successful profiler run [#39414502490](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490) produced [artifact #12261445684](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490/artifacts/12261445684): eight native `.ncu-repz` reports, Details/Raw CSV exports, logs, and **seven genuine 1936×1048 Nsight Compute GUI screenshots**. One screenshot remains open: the longer-batched LOAD_LATENCY=4 window returned `GetWindowRect failed`.
- The latest completed native validation on current kernel code is [#39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438) (commit `bf5a6233`): **48/48 core GPU tests**, **3/3 GQA/MQA and packed-varlen tests**, all **29 numerical case/tile combinations**, CLI-vs-NumPy oracle, three benchmark sessions, the correctness-gated load-latency sweep, and two new native Nsight Compute captures. [Artifact #12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195) (194,711 bytes; SHA-256 `e3eaf4ae1e88d2547d2034dfe333db86b70f69e2f2d6c71b0d67b7c8bdde0c32`) preserves that run's raw results/logs.
- In the latest three-session timed run, long D=128 tile 64×64 measured **0.267059 ms median / 16.49×**, and longer-batched D=128 tile 64×128 measured **0.915558 ms / 18.69×** versus the deliberately unfused FP32 CuPy baseline. The protocol used ten warmups, 30 CUDA-event samples, five launches per sample and three order-rotated sessions. These are not comparisons against FlashAttention or optimized PyTorch SDPA.
- The latest hint sweep passed accuracy-before-timing for all 36 trials. For medium D=128 / 64×64, **LOAD_LATENCY=10 measured 0.053248 ms vs 0.058342 ms default (8.73% faster)**, with 0.53% across-session range for both candidate and default; it clears the script's 1.00% noise-aware threshold. For longer-batched D=128 / 64×128, the best candidate was only 0.16% faster while observed default session range implied a 5.88% threshold; retain the compiler default there. Keep **LOAD_LATENCY=0 as the general default** until a matched NCU comparison of medium hint 10 is inspected; a dedicated NCU workflow adding that capture is queued.
- The fresh PyTorch run [#39422402415](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402415) failed all four autograd batch/head cases at `torch.bmm` with `CUBLAS_STATUS_INVALID_VALUE`. A diagnostic test in run [#39423190840](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840) reproduced the error in a **plain PyTorch CUDA GEMM before the cuTile bridge runs**, so this is not currently evidence of a cuTile-kernel defect. Importantly, that run's [artifact #12263887676](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840/artifacts/12263887676) also contains `cutile-accuracy-after-torch.json`: all 29 legal cuTile case/tile configurations still passed the independent NumPy tolerance check with the CUDA 13.0 PyTorch wheel present. This separates the observed plain-GEMM/cuBLAS failure from the already-passing cuTile forward kernel. The model-block test's `CUBLAS_STATUS_NOT_INITIALIZED` followed the earlier GEMM failure. A separate project-local CUDA 13.2 PyTorch wheel probe is queued in run [#39423804493](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423804493); it does not overwrite the primary venv. **Dispatch check (2026-10-20 13:29 UTC):** that probe, the fresh core regression [#39423691972](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423691972), and the matched hint-10 profile [#39423368441](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423368441) were all still queued with `runner_id=0`. The Windows workflows require `[self-hosted, windows, x64]`; no new GPU result is claimed until GitHub assigns a runner. No PyTorch integration, vendor-SDPA timing, or gradient pass is claimed yet.
- The earlier PowerShell startup failure [#39414502390](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502390) never reached GPU work; commit [da5b743](https://github.com/xemptynothing/gpu-nvidia/commit/da5b743d495abf44096772c71021b9c532517f1b) fixed the shell invocation. The genuine Nsight GUI capture helper now locates nested installs and records precise skipped-capture reasons.

## 5. Next completion gates

1. **Re-establish the core baseline on current main:** genuine cuTile smoke; full core correctness/edge tests; all 29 legal numerical case/tile configurations; CLI/NumPy oracle.
2. **Validate optional semantics on the GPU:** GQA/MQA head mapping, packed-sequence boundaries and outputs, and PyTorch forward/gradient-reference agreement.
3. **Measure before promoting:** PyTorch SDPA backend/op identity and fair timing; three-session autotuner stability; load-latency hint sweeps with per-candidate accuracy gates.
4. **Profile comparable builds:** preserve raw Nsight reports and exports for identical workloads, verify resource changes, and do not substitute profiler heuristics for elapsed-time measurements.
5. **Update the evidence ledger:** record run/artifact links, exact environment, commands, medians/P90/session variation, accuracy maxima and explicit keep/reject decisions. Update this two-pager after every material result.

## Reproduction and reading order

Start with the [README](../README.md), then this two-pager, the [complete experiment report](../Report.md), the [test matrix](TEST_MATRIX.md), and the [execution evidence index](EXECUTION_EVIDENCE_INDEX.md). The detailed phone-friendly guide is available at [CUTILE_ATTENTION_PROJECT_GUIDE_PHONE.pdf](CUTILE_ATTENTION_PROJECT_GUIDE_PHONE.pdf).

Core results remain the baseline evidence. New features become validated only after the corresponding native execution and output artifacts have been inspected.
