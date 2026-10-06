# Current Result Snapshot

Last known validated state of main:

| Area | Result |
|---|---|
| CPU/reference tests on self-hosted A100 runner | PASS |
| Required D coverage | 32 / 64 / 128 |
| Independent online-softmax reference | PASS |
| I/O and launch-contract tests | PASS |
| Self-hosted A100 runner routing | PASS |
| A100 detected | A100-SXM4-40GB, SM80 |
| Current node driver | R550.90.07 |
| Host driver upgrade from runner | NOT available: non-root; sudo requires credentials |
| Native current cuTile runtime | BLOCKED by driver prerequisite |
| cuda-compat-13-4 user-space experiment | Reached CUDA error 803 |
| Simple CUDA baseline | Implemented; GPU execution pending supported runtime |
| PyTorch SDPA reference | Implemented; optional |
| Benchmark harness | Implemented |
| Nsight target | Implemented |
| Raw benchmark artifacts | Pending first supported GPU execution |
| Driver-access audit | PASS: read-only diagnostics complete |

## Interpretation

A blocked cuTile runtime is an environment result, not a kernel-correctness result. The project must not report the fused kernel as correct or performant until it actually executes on a supported A100/cuTile environment.

Prior test and experiment logs remain in docs/PROGRESS.md.
