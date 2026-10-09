# Documentation index

Start here for the project's current state and evidence.

## Read in this order

1. [Experiment report](EXPERIMENT_REPORT.md) — concise end-to-end history of approaches, tests, benchmark results, failures, environment experiments, and decisions.
2. [Testing contract](TESTING_CONTRACT.md) — correctness oracle, tolerances, invariants, and regression rules.
3. [Test matrix](TEST_MATRIX.md) — test categories, coverage, and current pass/skip/block status.
4. [CUDA variants](CUDA_VARIANTS.md) — implementation-level differences between ordinary-CUDA versions.
5. [Baseline ladder](BASELINE_LADDER.md) — role of each reference and implementation.
6. [Environment](ENVIRONMENT.md) — A100/R550 constraints, compatibility findings, and profiling access.
7. [Progress log](PROGRESS.md) — dated chronological notes.
8. [Results storage conventions](../results/README.md) and [current status snapshot](../results/STATUS.md).

## One-line status

**The ordinary-CUDA A100 surrogate has passed its broad regression suite and isolated v4/v5 correctness tests. Actual cuTile execution remains blocked by the R550 driver prerequisite. Nsight Compute hardware counters are permission-blocked.** Native-CUDA performance measurements are surrogate results, not cuTile results.
