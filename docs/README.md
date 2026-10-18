# Documentation index

Start here for the project's current state and evidence.

## Read in this order

1. [TODO checklist](TODO_CHECKLIST.md) — completed gates, current blockers, prioritized next tasks, and definition of done.
2. [Local Windows GPU guide](LOCAL_GPU_WINDOWS.md) — RTX 4060 inventory, project-local Python/CuPy/cuTile setup, genuine smoke test, and safe next gates.
3. [Dockerized RTX 4060 Actions runner](CUTILE_DOCKER_RUNNER.md) — ephemeral GPU runner, host prerequisites, isolation choices, and one-command workflow dispatch.
4. [RTX 4060 actual-cuTile results](CUTILE_RTX4060_RESULTS_2026-10-19.md) — correctness, numerical error, repeat benchmarks and the profiler blocker.
5. [Experiment report](EXPERIMENT_REPORT.md) — concise end-to-end history of approaches, tests, benchmark results, failures, environment experiments, and decisions.
6. [Testing contract](TESTING_CONTRACT.md) — correctness oracle, tolerances, invariants, and regression rules.
7. [Test matrix](TEST_MATRIX.md) — test categories, coverage, and current pass/skip/block status.
8. [CUDA variants](CUDA_VARIANTS.md) — implementation-level differences between ordinary-CUDA versions.
9. [Baseline ladder](BASELINE_LADDER.md) — role of each reference and implementation.
10. [Environment](ENVIRONMENT.md) — A100/R550 constraints, compatibility findings, and profiling access.
11. [Admin checklist](ADMIN_CHECKLIST.md) — external cluster-side prerequisites.
12. [Progress log](PROGRESS.md) — dated chronological notes.
13. [Results storage conventions](../results/README.md) and [current status snapshot](../results/STATUS.md).

## One-line status

**Actual cuTile causal attention is validated and benchmarked on the RTX 4060.** The latest run passed 48 required tests, 29 accuracy diagnostics, and CLI/reference validation; three five-tile sessions are archived. Two actual-cuTile Nsight Compute profile reports remain outstanding because the CLI was absent from the project subtree and no download has been authorized.
