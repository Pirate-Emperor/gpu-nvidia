# Documentation index

Start here for the project's current state and evidence.

## Read in this order

1. [TODO checklist](TODO_CHECKLIST.md) — completed gates, current blockers, prioritized next tasks, and definition of done.
2. [Experiment report](EXPERIMENT_REPORT.md) — concise end-to-end history of approaches, tests, benchmark results, failures, environment experiments, and decisions.
3. [Testing contract](TESTING_CONTRACT.md) — correctness oracle, tolerances, invariants, and regression rules.
4. [Test matrix](TEST_MATRIX.md) — test categories, coverage, and current pass/skip/block status.
5. [CUDA variants](CUDA_VARIANTS.md) — implementation-level differences between ordinary-CUDA versions.
6. [Baseline ladder](BASELINE_LADDER.md) — role of each reference and implementation.
7. [Environment](ENVIRONMENT.md) — A100/R550 constraints, compatibility findings, and profiling access.
8. [Admin checklist](ADMIN_CHECKLIST.md) — external cluster-side prerequisites.
9. [Progress log](PROGRESS.md) — dated chronological notes.
10. [Results storage conventions](../results/README.md) and [current status snapshot](../results/STATUS.md).

## One-line status

**The ordinary-CUDA A100 surrogate has passed its broad regression suite and isolated v4/v5 correctness tests. Actual cuTile execution remains blocked by the R550 driver prerequisite. Nsight Compute hardware counters are permission-blocked.** Native-CUDA performance measurements are surrogate results, not cuTile results.
