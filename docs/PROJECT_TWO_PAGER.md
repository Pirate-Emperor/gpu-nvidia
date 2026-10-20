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

The strongest recorded result was the longer batched D=128 workload with a 64×128 tile: **0.919347 ms median**, **0.924795 ms P90**, and **18.62×** against the deliberately unfused FP32 CuPy baseline. This is not a speedup against FlashAttention or vendor-optimized PyTorch SDPA. The longer D=128 workload with 64×64 and 64×128 was effectively tied (about 0.08% difference), so **64×64 remains the general default** and 64×128 is workload-specific, not a universal promotion.

The NCU snapshots show a resource-constrained kernel with low achieved occupancy and many cycles without an eligible warp. Nsight's local speedup/latency estimates are heuristics, not measured speedups. The DRAM-load-latency hint experiment is only promoted if repeated timing, accuracy and profiler evidence support it.

## 4. Optional paths: implemented is not the same as validated

The current source also contains GQA/MQA launch support, a packed-variable-length wrapper, a PyTorch autograd/model bridge, a correctness-gated tile autotuner, and a cuTile DRAM-load-latency hint sweep. These were added after the last fully validated core GPU snapshot.

**Fresh native status — 20 October 2026 (Windows RTX 4060):**

- The native cuTile Nsight Compute job [run #39403495373](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403495373) completed successfully. Its [artifact #12256589808](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403495373/artifacts/12256589808) contains eight real `.ncu-repz` reports, matching Details/Raw CSV exports, and per-profile logs. These include base profiles, scheduler/warp-state profiles, and load-latency-hint 4/8 captures for both D=128 workloads.
- A separate native rerun [#39403488341](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403488341) passed the PyTorch/CuPy/cuTile import and CUDA-availability check, genuine cuTile smoke, required core GPU correctness tests, the 29-case/tile accuracy matrix, and CLI verification against the independent NumPy oracle. The full native-validation workflow [#39414502390](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502390) was also triggered but failed before package installation or any GPU test: the workflow's default PowerShell invocation was blocked by Windows execution policy. All its PowerShell steps are now being changed to explicit `-NoProfile -ExecutionPolicy Bypass -File {0}`; that workflow's fresh GPU pass remains pending. It then failed while collecting `tests/test_torch_integration.py`: `src/torch_integration.py` used `torch.autograd.Function` as a module-level base class without importing `torch` at module scope. This commit fixes that import defect; SDPA timing and later optional stages are **not yet claimed passed** until a new native run completes.
- The profiler's screenshot attempt produced **zero PNG screenshots**. Its manifest says none were captured; the profiler evidence is the native report/Details/CSV set, not GUI screenshots. The capture helper is being corrected to locate `ncu-ui.exe` across installation layouts and preserve explicit skip reasons.
- The previously recorded core benchmark remains ~0.92 ms median and ~18.5–18.6× versus the deliberately unfused FP32 CuPy baseline. That baseline is not FlashAttention or vendor-optimized PyTorch SDPA. Fresh hint 4/8 NCU durations are profiling observations only and must not be treated as ordinary benchmark results.

## 5. Next completion gates

1. **Re-establish the core baseline on current main:** genuine cuTile smoke; full core correctness/edge tests; all 29 legal numerical case/tile configurations; CLI/NumPy oracle.
2. **Validate optional semantics on the GPU:** GQA/MQA head mapping, packed-sequence boundaries and outputs, and PyTorch forward/gradient-reference agreement.
3. **Measure before promoting:** PyTorch SDPA backend/op identity and fair timing; three-session autotuner stability; load-latency hint sweeps with per-candidate accuracy gates.
4. **Profile comparable builds:** preserve raw Nsight reports and exports for identical workloads, verify resource changes, and do not substitute profiler heuristics for elapsed-time measurements.
5. **Update the evidence ledger:** record run/artifact links, exact environment, commands, medians/P90/session variation, accuracy maxima and explicit keep/reject decisions. Update this two-pager after every material result.

## Reproduction and reading order

Start with the [README](../README.md), then this two-pager, the [complete experiment report](../Report.md), the [test matrix](TEST_MATRIX.md), and the [execution evidence index](EXECUTION_EVIDENCE_INDEX.md). The detailed phone-friendly guide is available at [CUTILE_ATTENTION_PROJECT_GUIDE_PHONE.pdf](CUTILE_ATTENTION_PROJECT_GUIDE_PHONE.pdf).

Core results remain the baseline evidence. New features become validated only after the corresponding native execution and output artifacts have been inspected.
