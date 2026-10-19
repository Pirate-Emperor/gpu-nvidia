# Execution and Artifact Evidence Index

This file intentionally contains GitHub Actions links, workflow/runner diagnostics and artifact references that are excluded from the technical narrative in root Report.md. It is an execution record, not a substitute for the code, raw measurements or profiler exports.

## Core native RTX 4060 validation

- Full native validation: [run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223).
- Core test/benchmark artifact: [artifact #12149980979](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979).
- Observed successful gates: genuine cuTile smoke; 48 required attention tests; 29 numerical case/tile configurations with zero mixed-tolerance violations; CLI against independent NumPy; three controlled benchmark sessions.
- Hardware recorded by the run: RTX 4060 / SM89, Windows x64, driver 616.92, Python 3.12.10, CuPy 14.2.0, cuda-tile 1.6.0.

## Native Nsight Compute evidence

- Profiler workflow: [run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066).
- Profiler artifact: [artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).
- Artifact content: four compressed native NCU reports, four Details exports, four Raw CSV exports, plus capture/status logs.
- Captured workloads: medium D=128 / tile 64×64 and long-batched D=128 / tile 64×128; each has a basic resource/throughput profile and SchedulerStats/WarpStateStats profile.
- GPU counter collection succeeded on the RTX 4060 without ERR_NVGPUCTRPERM.
- The source report and capture helper for authentic GUI screenshots are root Report.md and scripts/capture_ncu_screenshots.py. Export-derived SVG plots are committed under docs/assets/report/.

## CPU/reference CI and test limitations

- Latest CPU/reference CI at the time of this index: [run #39249828521](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39249828521): 764 passed, 68 skipped, 0 failed.
- GPU-specific cases skipped in CPU-only CI when CuPy or CUDA is absent. Use the native RTX 4060 run as the evidence for real GPU execution.
- Further documentation-only CI runs are useful repository checks but should not be misreported as GPU validation.

## Fresh-run infrastructure failures (excluded from Report.md)

A deliberate final native rerun was started after the report/checklist update. It did not reach the kernel or test commands.

- [Run #39234069849](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39234069849), attempt 1: failed before checkout while resolving/download metadata for a GitHub Action. Logs reported “Failed to resolve action download info” after network retries.
- The same run, attempt 2: setup completed but the checkout step failed with a Git/TLS transport error, SEC_E_MESSAGE_ALTERED (0x8009030F), while fetching source. All GPU/test steps were skipped; this is not evidence of a kernel failure.
- Other native GPU/NCU jobs have at times remained queued while the self-hosted runner is unavailable or not accepting jobs. Those statuses are operational limitations, not test outcomes.

These runner errors are kept here solely for traceability and must not be described as CUDA kernel errors.

## Approved PyTorch SDPA / optional objective experiment

- Existing workflow: [PyTorch SDPA venv benchmark](https://github.com/xemptynothing/gpu-nvidia/blob/main/.github/workflows/torch-sdpa-preflight.yml).
- It is intended to install CUDA-enabled PyTorch only into the project-local .venv-cutile-windows environment, validate imports/CUDA availability and run:
  - scripts/benchmark_torch_sdpa.py
  - tests/test_torch_integration.py
  - benchmarks/benchmark_torch_model_integration.py
  - tests/test_gpu_optional_attention_objectives.py
  - scripts/autotune_cutile_tiles.py
- Evidence artifact will be added here once an actual native execution run finishes. A dry-run resolver result is not a completed install, and an installed package without CUDA or passing numerical tests does not qualify as an SDPA comparison.

## Report and image source-of-truth

- Canonical technical report: [Report.md](../Report.md).
- Rubric/checklist: [COURSE_PROJECT_COMPLETION_CHECKLIST.md](COURSE_PROJECT_COMPLETION_CHECKLIST.md).
- Profile interpretation: [CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md).
- Detailed benchmark snapshot: [CUTILE_RTX4060_RESULTS_2026-10-19.md](CUTILE_RTX4060_RESULTS_2026-10-19.md).
- The technical report contains external citations and project-relative figures, but deliberately omits action-run URLs and file checksums.


## Current approved rerun status — 19 October 2026

- PyTorch SDPA / optional experiment: [run #39252195780](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39252195780), currently queued on the self-hosted Windows runner at the latest status check. This run is on the updated workflow that fetches exact source files via the GitHub Contents API, records the project venv before/after installation, reruns smoke/core correctness/numerical/CLI checks, and then runs SDPA, model integration, optional tests, autotuning and load-latency hints.
- Authentic NCU screenshot / latency-hint captures: [run #39252236513](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39252236513), currently queued. The updated workflow fetches its pinned profiling and screenshot-helper source via Contents API rather than requiring a fresh git checkout. The screenshot step must create actual PNGs and a manifest; if the runner has no interactive desktop, it should record skipped captures rather than fabricate them.
- CPU/reference CI after those workflow changes: [run #39252236625](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39252236625) completed successfully. The CI is CPU/reference-only; GPU tests skipping for missing CuPy are not counted as native GPU results.
- The obsolete duplicate framework experiment workflow from earlier in the session was removed from main and should not be used. Its old run #39249828524 is historical/queued at an older commit; the canonical experiment is the existing PyTorch SDPA venv benchmark workflow.

The native jobs being queued are **not** evidence that the new SDPA install, screenshots, or optional objectives have passed. Add the experiment’s actual logs and result artifact after it has run.
