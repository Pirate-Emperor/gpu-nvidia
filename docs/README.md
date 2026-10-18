# Documentation index

Start here for the project's current state and evidence.

## Read in this order

1. [TODO checklist](TODO_CHECKLIST.md) — completed gates, current blockers, prioritized next tasks, and definition of done.
2. [Local Windows GPU guide](LOCAL_GPU_WINDOWS.md) — RTX 4060 inventory, project-local Python/CuPy/cuTile setup, genuine smoke test, and safe next gates.
3. [Experiment report](EXPERIMENT_REPORT.md) — concise end-to-end history of approaches, tests, benchmark results, failures, environment experiments, and decisions.
4. [Testing contract](TESTING_CONTRACT.md) — correctness oracle, tolerances, invariants, and regression rules.
5. [Test matrix](TEST_MATRIX.md) — test categories, coverage, and current pass/skip/block status.
6. [CUDA variants](CUDA_VARIANTS.md) — implementation-level differences between ordinary-CUDA versions.
7. [Baseline ladder](BASELINE_LADDER.md) — role of each reference and implementation.
8. [Environment](ENVIRONMENT.md) — A100/R550 constraints, compatibility findings, and profiling access.
9. [Admin checklist](ADMIN_CHECKLIST.md) — external cluster-side prerequisites.
10. [Progress log](PROGRESS.md) — dated chronological notes.
11. [Results storage conventions](../results/README.md) and [current status snapshot](../results/STATUS.md).

## One-line status

**Actual cuTile has not yet run.** The existing A100/R550 route is blocked and remains an ordinary-CUDA control. The local RTX 4060 (CC 8.9, driver 616.92, Windows x64) fits NVIDIA's published support matrix on paper; per-user CPython 3.12, the project venv, and a genuine cuTile smoke test are next. Detailed profiler access remains unverified. Native-CUDA measurements are surrogate results, not cuTile results.
