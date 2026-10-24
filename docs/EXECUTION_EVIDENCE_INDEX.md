# Execution evidence

This page links the main runs and saved files. Older failures remain in the log for context; a setup or runner failure is not a kernel result.

## Core GPU validation

- Last complete native cuTile run: [#39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), commit bf5a623.
- It passed the 48 core tests, three selected extension tests, 29 numerical configurations, CLI check and three timing sessions.
- The cuTile kernel was compacted after this run. Current main still needs a fresh native GPU pass.

## Timing and profiling

- [Benchmark and validation artifact #12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195).
- [Latest successful profiler run #39414502490](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490).
- [Profiler artifact #12261445684](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490/artifacts/12261445684): eight native reports, matching metric exports and seven genuine GUI screenshots.
- [Profiler analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) explains the collected counters and their limits.

## Current follow-up runs

- Current-main native validation: [#40118382542](https://github.com/xemptynothing/gpu-nvidia/actions/runs/40118382542).
- Current-main detailed NCU capture: [#40118383063](https://github.com/xemptynothing/gpu-nvidia/actions/runs/40118383063).

Check the run status before quoting current-source results.

## Limits

The timing baseline is unfused FP32 CuPy, not vendor-optimised attention. Long-specialization first-call timing is incomplete because some kernels were already compiled. Detailed profiler counter collection can also be limited by the driver or by unavailable metrics.

Use Report.md for measured results and RUN_EVIDENCE.md for older workflow and runner problems.
