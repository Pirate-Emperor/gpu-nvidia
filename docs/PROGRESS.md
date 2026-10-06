# Project Progress Log

This file is the durable experiment log for the project. Record what was tested, what passed, what failed, and what remains blocked.

## 2026-10-05 / 2026-10-06 — Foundation

### Repository foundation
- Added an independent FP32 NumPy mathematical reference.
- Added a tiled online-softmax NumPy reference.
- Added a deliberately unfused CuPy GPU baseline.
- Added the first cuTile fused causal-attention kernel.
- Added required launch-contract validation.
- Added deterministic input generation and exact NPZ/NPY I/O.
- Added correctness/contract/I/O/generator/online-reference tests.
- Added a frozen representative benchmark matrix.

### CPU validation
- The self-hosted A100 runner successfully executed the CPU/reference suite.
- The corrected suite reached 75 passing tests.
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
- Added a simple scalar CUDA baseline.
- Added an optional PyTorch SDPA reference.
- Added baseline correctness tests.
- Added a project-structure test that freezes required deliverables.
- Added a read-only driver-access audit workflow.
- Added benchmark and Nsight workflows that are ready for a supported A100/cuTile runtime.
- Added result-storage conventions and a persistent status snapshot.

## Baseline ladder added

The repository now contains:
1. NumPy FP32 full attention.
2. NumPy tiled online-softmax.
3. Unfused CuPy attention.
4. Simple scalar CUDA attention.
5. Optional PyTorch SDPA reference.
6. cuTile fused attention.

Each GPU backend has a correctness-test path. Exploratory baselines are not confused with the required cuTile deliverable.

## Next execution gate

1. Determine whether another existing A100 node has R580+.
2. Check whether the cluster permits a system driver upgrade or provides a supported compatibility environment.
3. Once a supported runtime is available, execute the full cuTile correctness matrix.
4. Capture at least two Nsight Compute profiles.
5. Run >=30-iteration benchmark sessions and preserve raw JSON/CSV.
6. Optimize only after correctness and profiling evidence exist.

## Experiment policy

For each material kernel change:
1. preserve/update the relevant tests;
2. record the exact environment and configuration;
3. record successful and unsuccessful outcomes;
4. re-run correctness before claiming a performance improvement;
5. preserve benchmark/profiler artifacts instead of replacing prior runs.
