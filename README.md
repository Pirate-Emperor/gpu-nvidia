# Fused Causal Attention with NVIDIA cuTile

IIT Madras GPU Programming course project: implement and optimize causal scaled dot-product self-attention as a fused NVIDIA cuTile GPU kernel, targeted at NVIDIA A100/SM80.

## Goal

The required kernel must:

- consume FP16 Q/K/V;
- accumulate in FP32;
- implement causal self-attention only;
- fuse score computation, scaling, causal masking, stable online softmax, and V accumulation;
- avoid materializing the complete score/probability matrix in global memory;
- support the required head dimensions D = 32, 64, 128;
- require tile-divisible sequence lengths for the selected tile sizes.

The course proposal additionally requires at least two tile configurations, deterministic correctness tests, an unfused baseline, a framework/vendor performance reference, >=30 measured iterations, separate JIT reporting, and profiler evidence.

## Repository layout

src/reference.py
  Independent FP32 NumPy mathematical oracle.

src/online_reference.py
  Independent tiled online-softmax recurrence.

src/baseline.py
  Deliberately unfused CuPy GPU baseline that materializes score/probability tensors.

src/naive_cuda_attention.py
  Simple scalar CUDA baseline. It is intentionally inefficient and exists to establish a basic CUDA performance floor.

src/torch_reference.py
  Optional PyTorch SDPA reference/performance backend.

src/cutile_attention.py
  Main cuTile fused implementation.

tests/
  Correctness and contract tests.

benchmarks/benchmark_attention.py
  Fused-vs-unfused benchmark using GPU events.

benchmarks/profile_fused.py
  Warmed single-launch target for Nsight Compute.

docs/BASELINE_LADDER.md
  Comparison strategy.

docs/TESTING_CONTRACT.md
  Frozen correctness contract.

docs/PROGRESS.md
  Persistent experiment log.

results/
  Raw benchmark and profiler artifacts.

## CPU validation

CPU/reference tests need only NumPy and pytest:

    python -m pip install -r requirements-dev.txt
    pytest -q -m "not gpu"

The CPU suite does not validate CUDA execution.

## GPU environment

Internet-dependent installation on the project runner must go through the root-level core_config:

    python3 scripts/with_core_config.py <command> ...

This imports the user's root-level core_config and preserves its configured proxy environment without storing proxy credentials in the repository.

## Required CLI

    python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile

## Correctness matrix

The GPU target is checked against the independent FP32 NumPy oracle with a documented mixed tolerance.

Coverage includes:

- D = 32, 64, 128.
- Short, medium, and long sequences.
- Multiple head counts and batch sizes.
- Multiple tile configurations.
- Small and large score magnitudes.
- Causal future-token invariance.
- Rejection of non-divisible sequence lengths.
- NaN/Inf checks.

## Baseline ladder

1. NumPy FP32 oracle.
2. NumPy tiled online-softmax reference.
3. Unfused CuPy GPU baseline.
4. Simple scalar CUDA baseline.
5. Optional PyTorch SDPA reference.
6. Fused cuTile target.

This lets us distinguish mathematical correctness from GPU-kernel optimization.

## Performance method

Steady-state measurements use GPU events, warm-up iterations, >=30 measured iterations, median/P90 latency, speedup versus the deliberately unfused baseline, effective output throughput, explicit GPU/toolchain metadata, and separate cold/JIT timing.

Profiler runs use a dedicated single-kernel target so Nsight Compute measurements are not polluted by the entire benchmark harness.

## Current status

CPU/reference validation is passing on the self-hosted A100 runner.

The current A100 node reports driver R550.90.07. Current NVIDIA cuTile releases require a newer driver, so the production cuTile workflow intentionally stops at the environment gate rather than claiming unsupported kernel execution.

A separate forward-compatibility experiment downloaded NVIDIA's cuda-compat-13-4 package without root and tested it, but CUDA reported error 803 (system driver mismatch). See docs/PROGRESS.md.

## Course-scope boundary

The following are exploratory only and should be postponed until the required path is correct, benchmarked, and profiled:

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache, packed variable-length sequences, dropout, partial edge tiles, and alternative DSL implementations.
