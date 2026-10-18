# Test Matrix and Validation Ledger

This file records test coverage and the latest known execution state. For the consolidated approaches, exact outcomes, and measured comparisons, see [docs/EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).

## Latest historical run records (handoff checked 2026-10-18; Actions details unavailable in this continuation)

| Workflow | Result | Important boundary |
|---|---|---|
| [A100 native CUDA #44](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37756646415) | 133 passed, 1 skipped | Actual ordinary CUDA kernels on A100/R550; no cuTile claim. |
| [A100 full non-cuTile #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082) | 907 passed, 52 skipped | Broad non-cuTile regression. |
| [CPU/reference #340](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056390931) | 763 passed, 68 skipped | Latest successful CPU/reference suite, on pre-handoff main SHA afe530b; GPU-only tests skip in hosted CPU environment. |
| [v4 isolated A100 tests](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37760982099) | 30 passed | Experimental variant; not promoted. |
| [v5 isolated A100 tests](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/37761585734) | 30 passed | Experimental half2 variant; performance inconclusive. |
| [v5 paired benchmark #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163) | Correctness 30/30 + 3 paired repeats/case/tile | Order alternated; v5 remains unpromoted after mixed/noisy results. |
| [PTXAS resource report #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) | v1/v2/v3/current/v4/v5 compiled; zero spills | v5 added to resource comparison; compile-time evidence only. |
| [Nsight Systems timeline #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950) | Two warmed surrogate `.nsys-rep` traces plus CUDA API/kernel summaries uploaded ([artifact](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954)) | CUDA API/kernel timeline only; not actual-cuTile or Nsight Compute counters. |


**Actions evidence access note (2026-10-18 continuation):** the connected Actions run-list endpoint returned zero visible runs; direct run-detail, job and artifact requests for the handoff IDs returned 404, and the current handoff commit had no visible combined status. This does not prove the historical runs do not exist. Historical counts below remain from the prior handoff and were not independently re-audited through this connection.

## 1. Reference / CPU tests

| Test class | Coverage | Expected property | Latest state |
|---|---|---|---|
| Shape/dtype contract | rank-4, FP16, contiguous, matching B/H/N/D | invalid inputs rejected | PASS |
| Required head dimensions | D=32, 64, 128 | all required dimensions execute in oracle | PASS |
| Causal semantics | future K/V mutations | earlier outputs unchanged | PASS |
| Query locality | changing q[r] | earlier rows unchanged | PASS |
| N=1 | single-token attention | output exactly V0 | PASS |
| Zero logits | uniform causal probabilities | prefix mean of V | PASS / expanded |
| Constant V | arbitrary Q/K | output exactly constant V | PASS / expanded |
| Extreme logits | large finite scores | no NaN/Inf; stable softmax | PASS / expanded |
| Convex-hull invariant | arbitrary finite values | each output lies within prefix V range | PASS / expanded |
| Batch/head isolation | B>1, H>1 | independent per-head behavior | PASS / expanded |
| Online-softmax recurrence | multiple legal tile sizes | matches independent FP32 oracle | PASS |
| Numerical metric tests | abs/rel mixed tolerance | boundary behavior verified | PASS |
| I/O contract | exact NPZ keys / NPY output | reproducible CLI data path | PASS |
| Generator contract | deterministic seeded inputs | valid required configurations | PASS |
| Project structure | required source/tests/docs/workflows | deliverables remain present | PASS |

The self-hosted A100 runner has previously completed the CPU/reference suite successfully; the repository records the exact suite result in `docs/PROGRESS.md`.

## 2. GPU cuTile correctness matrix

The A100 GPU suite is intentionally broader than the minimum course matrix.

### Frozen representative cases

- `short_d32`: B=1, H=1, N=64, D=32
- `short_d64`: B=1, H=1, N=64, D=64
- `short_multihead_d64`: B=1, H=4, N=128, D=64
- `medium_d64`: B=1, H=8, N=256, D=64
- `medium_d128`: B=1, H=8, N=512, D=128
- `long_d128`: B=1, H=16, N=1024, D=128
- `longer_batched_d128`: B=2, H=8, N=2048, D=128

### Required GPU checks

| Check | Coverage | Current state |
|---|---|---|
| Independent FP32 reference comparison | all mandatory cases | BLOCKED by R550 runtime gate |
| D=32 / 64 / 128 | mandatory cases + targeted edge tests | BLOCKED |
| Two tile configurations | 64x64 and 128x128 | BLOCKED |
| Asymmetric legal tile configurations | targeted 32x64 and 64x32 | BLOCKED |
| Small-logit stability | dedicated test | BLOCKED |
| Large-logit stability | dedicated test | BLOCKED |
| NaN / Inf rejection before launch | dedicated validation tests | BLOCKED |
| Non-contiguous input rejection | dedicated validation test | BLOCKED |
| Causal future-V invariance | dedicated test | BLOCKED |
| Causal future-K invariance | dedicated test | BLOCKED |
| Constant-V exactness | D=32/64/128 | BLOCKED |
| Zero-logit prefix means | D=64 | BLOCKED |
| Tile-to-tile numerical agreement | 64x64 vs 128x128 | BLOCKED |
| Multiple B/H | B=2 and multi-head cases | BLOCKED |
| Custom block-row CUDA | D=32/64/128 + causal/edge/stability cases | PASS; A100 native run #44 |
| Custom warp-row CUDA | D=32/64/128 + causal/edge/stability cases | PASS; A100 native run #44 |
| Cross-implementation agreement | block-row vs warp-row | PASS; native CUDA suite |
| Shared-memory tiled CUDA (current) | tile_k=16/32/64, D=32/64/128, causality/stability/long sequences | PASS; native run #44 and broad run #52 |
| Shared-tile v4 experimental | differential tests, future-K/V invariance, long-sequence smoke | PASS; 30/30 in dedicated workflow |
| Shared-tile v5 experimental | differential tests, future-K/V invariance, long-sequence smoke, tile_k=16/32/64 | PASS; 30/30 in dedicated workflow; performance inconclusive |

## 3. GPU baseline ladder

Every executable GPU implementation has a separate correctness path:

1. NumPy FP32 mathematical oracle.
2. Tiled NumPy online-softmax oracle.
3. Unfused CuPy score/probability materialization baseline.
4. Scalar CUDA baseline.
5. Custom block-row fused CUDA.
6. Custom warp-row fused CUDA.
7. Optional PyTorch SDPA reference.
8. Fused cuTile kernel.

The two custom CUDA variants are deliberately independent implementations of
the fused algorithm. They are cross-checked against the mathematical oracle,
the online recurrence, and each other before they are used as a reference for
cuTile.

## 4. Performance runs

Native-CUDA surrogate timings have been collected on the A100 using CUDA events, 10 warmups, 30 measured iterations, median/P90, and raw JSON/CSV artifacts. The tile ladder also measures first compilation separately. The current-vs-v5 experiment uses three paired repeats with order alternating by repeat; see [v5 benchmark #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163). The v5 result remains inconclusive and it is not promoted. See [the experiment report](EXPERIMENT_REPORT.md) for medians and the explicit warning about run-to-run timing variability.

Actual cuTile performance remains blocked until the kernel executes on a supported driver/runtime. Nsight Compute hardware profiling is additionally blocked by `ERR_NVGPUCTRPERM`; current profiling claims are therefore limited to successful compiler/PTXAS resource reports, not hardware-counter data.

Still required for final course submission:
- Actual cuTile execution and correctness on the inventoried local RTX 4060 after CPython 3.12 and the real cuTile smoke test pass. The current R550 A100 is not the actual-cuTile route.
- Final representative benchmark matrix and cuTile-vs-unfused/framework comparison.
- At least two successful hardware profiles with counter access.
- Final deliverable/report after those runs.

## 5. Environment gate

**Existing hosted/self-hosted control runner:** NVIDIA A100-SXM4-40GB, SM80, driver R550.90.07. This runner is below NVIDIA's R580+ cuTile Python requirement and remains for ordinary-CUDA surrogate validation only. Prior forward-compatibility attempts did not establish actual cuTile execution.

**Local actual-cuTile candidate (inventory 2026-10-18):** GeForce RTX 4060, CC 8.9, driver 616.92, Windows x64 build 26300, 8,188 MiB VRAM. These GPU/driver/OS facts fit NVIDIA's documented support matrix on paper. Python and system NVCC are not installed; native Windows venv setup and the genuine cuTile smoke test are pending. Do not convert a skipped GPU test into a pass. See [local GPU guide](LOCAL_GPU_WINDOWS.md).
## 6. Dual-device hardware validation

The actual runner exposes two full A100-SXM4-40GB GPUs through `CUDA_VISIBLE_DEVICES=0,1`.
The native-CUDA suite therefore includes:
- one CUDA arithmetic smoke test on device 0 and device 1;
- one fused-attention execution on device 0 and device 1;
- a cross-device output agreement test using identical inputs.

This is hardware validation, not a cuTile result. The cuTile workflow remains a
separate R580-gated execution path.

## 7. Compilation backends

The ordinary CUDA implementations are compiled through CuPy's `nvcc` backend
with `-arch=sm_80`, using the host's CUDA 12.4 compiler. This intentionally
avoids the current NVRTC header-discovery failure on the R550 node.
