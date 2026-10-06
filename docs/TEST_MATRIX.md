# Test Matrix and Validation Ledger

This file is the human-readable record of what the project tests, why each class exists, and the latest known execution state.

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
| Custom block-row CUDA | D=32/64/128 + causal/edge/stability cases | NEW; pending GPU CI |
| Custom warp-row CUDA | D=32/64/128 + causal/edge/stability cases | NEW; pending GPU CI |
| Cross-implementation agreement | block-row vs warp-row | NEW; pending GPU CI |
| Shared-memory tiled CUDA | tile_k=32/64, D=32/64/128 | NEW; pending GPU CI |
| Three-way fused agreement | block-row vs warp-row vs shared-tile | NEW; pending GPU CI |

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

The performance suite will use CUDA events, warmups, synchronized steady-state timing, median latency, P90 latency, and raw JSON/CSV preservation.

Required future runs:
- >=6 representative configurations.
- >=30 timed iterations per configuration.
- JIT/cold-start latency reported separately.
- fused vs unfused speedup.
- effective output throughput.
- Nsight Compute on >=2 representative configurations.

No GPU performance number is marked valid until the kernel executes successfully on a supported A100/cuTile environment.

## 5. Environment gate

Current runner: NVIDIA A100-SXM4-40GB, SM80, driver R550.90.07.

Current NVIDIA cuTile Python documentation requires R580+; the current user-space forward-compatibility experiment on this node produced CUDA error 803. Therefore GPU execution remains an environment gate rather than an untested result.