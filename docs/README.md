# Documentation index

Start here for the project's current state and evidence.

## Read in this order

1. [TODO checklist](TODO_CHECKLIST.md) — completed gates, current blockers, prioritized next tasks, and definition of done.
2. [Local Windows GPU guide](LOCAL_GPU_WINDOWS.md) — RTX 4060 inventory, project-local Python/CuPy/cuTile setup, genuine smoke test, and safe next gates.
3. [Dockerized RTX 4060 Actions runner](CUTILE_DOCKER_RUNNER.md) — ephemeral GPU runner, host prerequisites, isolation choices, and one-command workflow dispatch.
4. [Experiment report](EXPERIMENT_REPORT.md) — concise end-to-end history of approaches, tests, benchmark results, failures, environment experiments, and decisions.
5. [Testing contract](TESTING_CONTRACT.md) — correctness oracle, tolerances, invariants, and regression rules.
6. [Test matrix](TEST_MATRIX.md) — test categories, coverage, and current pass/skip/block status.
7. [CUDA variants](CUDA_VARIANTS.md) — implementation-level differences between ordinary-CUDA versions.
8. [Baseline ladder](BASELINE_LADDER.md) — role of each reference and implementation.
9. [Environment](ENVIRONMENT.md) — A100/R550 constraints, compatibility findings, and profiling access.
10. [Admin checklist](ADMIN_CHECKLIST.md) — external cluster-side prerequisites.
11. [Progress log](PROGRESS.md) — dated chronological notes.
12. [Results storage conventions](../results/README.md) and [current status snapshot](../results/STATUS.md).

## One-line status

**The genuine cuTile smoke test passed on the local RTX 4060**, but the causal-attention kernel remains unvalidated. The existing A100/R550 route remains an ordinary-CUDA control. Docker Desktop/WSL2 are still required before the new ephemeral GPU Actions runner can run the actual attention correctness matrix. Detailed profiler access remains unverified. Native-CUDA measurements are surrogate results, not cuTile results.
