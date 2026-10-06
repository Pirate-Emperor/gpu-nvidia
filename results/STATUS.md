# Current Result Snapshot

Last known validated state of main:

| Area | Result |
|---|---|
| CPU/reference tests on self-hosted A100 runner | PASS; 75 passed before latest edge-case expansion |
| Expanded CPU/reference suite | Added; pending CI execution of latest commit |
| Required D coverage | 32 / 64 / 128 |
| Independent online-softmax reference | PASS |
| I/O and launch-contract tests | PASS |
| Self-hosted A100 runner routing | PASS |
| A100 detected | A100-SXM4-40GB, SM80 |
| Current node driver | R550.90.07 |
| Host driver upgrade from runner | NOT available: non-root; sudo requires credentials |
| Native current cuTile runtime | BLOCKED by driver prerequisite |
| cuda-compat-13-4 user-space experiment | Reached CUDA error 803 |
| Scalar CUDA baseline | Implemented; native CUDA execution workflow added |
| Custom block-row fused CUDA | Implemented; native CUDA execution workflow added |
| Custom warp-row fused CUDA | Implemented; native CUDA execution workflow added |
| Custom shared-memory tiled CUDA | Implemented; native CUDA execution workflow added |
| Unfused CuPy baseline | Implemented; native CUDA execution workflow added |
| PyTorch SDPA reference | Implemented; optional |
| Benchmark harness | Implemented |
| Nsight target | Implemented |
| Raw benchmark artifacts | Pending first supported GPU execution |
| Detailed test matrix | Added at docs/TEST_MATRIX.md |
| Driver-access audit | PASS: read-only diagnostics complete |

## Interpretation

The project now has broader CPU/reference coverage and a driver-independent native-CUDA validation path. The native CUDA variants are implemented but their latest CI execution result is not yet available here. cuTile results remain unvalidated because the current A100 node is on R550. The repository must not report the cuTile kernel as correct or performant until it executes on a supported A100/cuTile environment.

Prior test and experiment logs remain in docs/PROGRESS.md.
