# Documentation index

Start with the [Project Two-Pager](PROJECT_TWO_PAGER.md) for the current concise overview and gates, then use [the complete experiment narrative](../Report.md) for the detailed assignment, test, benchmark, profiler, error and optional-objective record. [Run provenance](../RUN_EVIDENCE.md) is intentionally kept separate.

## Read in this order

0. [Project Two-Pager](PROJECT_TWO_PAGER.md) — compact current status, strongest recorded core result, implementation map and validation gates.
0. [Complete experiment narrative](../Report.md) — single authoritative record for the course scope, all core tests/results, all benchmark rows, profiler analysis/plots, project/CUDA/library errors, optional objectives and next actions.
0a. [Automation/run provenance](../RUN_EVIDENCE.md) — separate workflow/runner URLs and infrastructure failures; not part of the technical report.
0b. [Step-by-step mentor walkthrough](MENTOR_STEP_BY_STEP_WALKTHROUGH.md) — run smoke, the two GPU test files, CLI/oracle, 29 legal numerical configurations, full three-session tile experiment and load-latency hint sweep one stage at a time; Nsight Compute is deferred.
0c. [cuTile source walkthrough](CUTILE_KERNEL_WALKTHROUGH.md) — how the unfused baseline and fused kernel execute, online-softmax recurrence, rationale behind all 48 test executions, and the recorded experiment decisions.
0d. [Editable numerical configuration list](../tests/legal_attention_configs.json) — the workload shapes and tile candidates used for legal numerical comparisons.
0e. [Configuration registry editor and validation](../scripts/legal_config_registry.py) plus [registry tests](../tests/test_legal_config_registry.py) — checks shape/tile constraints and verifies that adding a workload or tile expands the numerical matrix.
0f. [Formal course submission report](SUBMISSION_REPORT_2026-10-22.md) — concise final results, numerical accuracy, performance comparisons, profiler metrics, and known limitations.

1. [Rubric-mapped course project checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md) — requirement-by-requirement closure, status, evidence and prioritized remaining work.
2. [Detailed course project report](COURSE_PROJECT_REPORT_2026-10-19.md) — algorithm derivation, correctness, performance, profiler interpretation, limitations and submission state.
3. [Project TODO checklist](TODO_CHECKLIST.md) — implementation gates, evidence and engineering priorities.
4. [RTX 4060 actual-cuTile results](CUTILE_RTX4060_RESULTS_2026-10-19.md) — correctness, numerical error, benchmark tables and canonical evidence links.
5. [Nsight Compute profile analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) — four real-cuTile profile captures, exports, resource/occupancy data, scheduler observations and limitations.
6. [Experiment report](EXPERIMENT_REPORT.md) — implementation variants, historical experiments, failures and decisions.
7. [Test matrix](TEST_MATRIX.md) and [testing contract](TESTING_CONTRACT.md) — coverage, reference tolerance and regression expectations.
8. [Baseline ladder](BASELINE_LADDER.md) and [CUDA variants](CUDA_VARIANTS.md) — role of each baseline and implementation family.
9. [Local Windows GPU guide](LOCAL_GPU_WINDOWS.md) and [runner guide](CUTILE_DOCKER_RUNNER.md) — validated native setup and optional isolation route.
10. [Environment](ENVIRONMENT.md) and [admin checklist](ADMIN_CHECKLIST.md) — driver/toolchain prerequisites and external runner constraints.
11. [Progress log](PROGRESS.md) — dated experiment record.
12. [Results storage conventions](../results/README.md) and [current status snapshot](../results/STATUS.md).

## Current evidence-based status

**Historical core GPU gates PASS; current main is pending a fresh native regression; assignment closure remains PARTIAL.** Native RTX 4060 run [#39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223) passed the cuTile smoke, 48 required attention tests, 29 numerical configurations, CLI/reference verification and three controlled benchmark sessions. Four actual-cuTile Nsight Compute reports and CSV/Details exports are preserved in [profile artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039). Since that baseline, optional GQA/MQA, packed-sequence, PyTorch bridge, autotuner and load-latency-hint paths were added, so the old pass does not automatically validate the current source. The latest CPU/reference CI passes; fresh native SDPA/core regression and NCU jobs are queued after a Windows PowerShell startup fix. Remaining rubric items are visible in the checklist: framework/vendor attention reference; clean-cache JIT times for long specializations or an approved exception; Week 3’s second profiler-guided improvement; and accurate author/team contribution details.

The most recent completed CPU/reference CI is documented separately in [RUN_EVIDENCE.md](../RUN_EVIDENCE.md); GPU-specific tests skip on CPU-only CI because CuPy is not installed. See [Report.md](../Report.md) for the currently verified GPU results and remaining optional-objective status. GPU-specific tests skip on that CPU-only runner because CuPy is not installed; the separate native RTX 4060 run is the GPU evidence.
