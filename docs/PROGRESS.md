# Project Progress Log

This file is the durable experiment log for the project. Record what was tested, what passed, what failed, and what remains blocked.

## 2026-10-05 / 2026-10-06 — Foundation

### Repository foundation
- Added an independent FP32 NumPy mathematical reference.
- Added a tiled online-softmax NumPy reference.
- Added a deliberately unfused CuPy GPU baseline.
- Added a scalar CUDA baseline.
- Added an optional PyTorch SDPA reference.
- Added the first cuTile fused causal-attention kernel.
- Added required launch-contract validation.
- Added deterministic input generation and exact NPZ/NPY I/O.
- Added correctness/contract/I/O/generator/online-reference tests.
- Added a frozen representative benchmark matrix.
- Added a persistent human-readable test/validation ledger at docs/TEST_MATRIX.md.

### Expanded test coverage
- Added randomized finite-output coverage across B/H/N and D=32/64/128.
- Added exact prefix-mean checks for zero logits.
- Added constant-V invariance checks across all required head dimensions.
- Added extreme finite-logit stability checks.
- Added convex-hull invariants for attention outputs.
- Added explicit batch/head isolation checks.
- Added additional current-query locality checks.
- Added GPU-only tests for asymmetric legal tile configurations, tile-to-tile agreement, causal future-K/V invariance, constant-V behavior, zero-logit prefix means, nonfinite input rejection, non-contiguous rejection, and unsupported/mismatched dimensions.

### CPU validation
- The self-hosted A100 runner successfully executed the CPU/reference suite.
- The corrected suite reached 75 passing tests before this latest edge-case expansion.
- The newly added CPU tests are preserved and will run automatically through the repository test workflow.
- One initial test failure was a test-shape assertion, not an attention implementation failure; it was fixed and re-run successfully.

### A100 environment
- The self-hosted runner selects an NVIDIA A100-SXM4-40GB node.
- The node reports 40 GiB and compute capability 8.0.
- Current driver is R550.90.07.
- Current NVIDIA cuTile Python releases require a newer driver, so the production cuTile workflow stops at an explicit environment gate.
- A read-only access audit reports UID/EUID 55012, not root; sudo exists but passwordless sudo is unavailable and noninteractive sudo policy inspection requires credentials.
- Conclusion: a host-driver upgrade cannot be performed by the GitHub runner account itself. It requires an administrator/privileged maintenance path, unless the cluster provides a separately authorized upgrade mechanism.

### Forward-compatibility experiment
- NVIDIA cuda-compat-13-4 was downloaded from the NVIDIA repository without root.
- The compatibility libraries were extracted into the job workspace.
- Native CUDA 13.x access failed with cudaErrorInsufficientDriver.
- Loading the extracted compatibility libraries produced cudaErrorSystemDriverMismatch (CUDA error 803).
- Therefore the current R550 node has not been validated as a cuTile runtime through that compatibility configuration.

### cuTile 1.2 experiment
- A branch attempted to test an early cuTile release associated with Ampere/SM80 support.
- The currently reachable Python package index does not expose cuda-tile 1.2.0, so that branch did not reach GPU execution.

## Engineering infrastructure added
- Added a project-structure test that freezes required deliverables.
- Added a read-only driver-access audit workflow.
- Added benchmark and Nsight workflows that are ready for a supported A100/cuTile runtime.
- Added result-storage conventions and a persistent status snapshot.

## Baseline ladder
1. NumPy FP32 full attention.
2. NumPy tiled online-softmax.
3. Unfused CuPy attention.
4. Simple scalar CUDA attention.
5. Optional PyTorch SDPA reference.
6. Fused cuTile attention.

Each executable GPU backend has an independent correctness-test path against the mathematical oracle.

## Next execution gate
1. Ask the cluster administrator for an R580+ driver path or another supported A100 node.
2. Run the expanded GPU correctness matrix.
3. Capture at least two Nsight Compute profiles.
4. Run >=30-iteration benchmark sessions and preserve raw JSON/CSV.
5. Optimize only after correctness and profiling evidence exist.

## Experiment policy
For each material kernel change:
1. preserve/update the relevant tests;
2. record the exact environment and configuration;
3. record successful and unsuccessful outcomes;
4. re-run correctness before claiming a performance improvement;
5. preserve benchmark/profiler artifacts instead of replacing prior runs.
