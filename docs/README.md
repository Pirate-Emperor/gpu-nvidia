# Documentation index

Read these first:

- [Repository README](../README.md) — current project state, benchmark interpretation, partial speedups and remaining gates.
- [Assignment brief audit](ASSIGNMENT_BRIEF_AUDIT.md) — direct mapping to the six-page `nvidia2-cutile.pdf`.
- [Course completion checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md) — assignment acceptance gates and current status.
- [Current run evidence — 2 November 2026](CURRENT_RUN_2026-11-02.md) — native RTX 4060 correctness, benchmarks, PyTorch SDPA, JIT, NCU comparison and open gates.
- [TODO checklist](TODO_CHECKLIST.md) — project work items.
- [Project two-pager](PROJECT_TWO_PAGER.md) — compact problem/kernel/evidence summary.
- [Detailed RTX 4060 results](CUTILE_RTX4060_RESULTS_2026-10-20.md) — 29-pair historical CuTile timing matrix and hint sweep.
- [Baseline ladder](BASELINE_LADDER.md) — separates NumPy references, unfused CuPy, CUDA controls and CuTile.
- [CUDA variants](CUDA_VARIANTS.md) — ordinary CUDA C++ variants v1–v5; they are not actual NVIDIA cuTile variants.
- [Kernel walkthrough](CUTILE_KERNEL_WALKTHROUGH.md) — implementation details.
- [Mentor walkthrough](MENTOR_STEP_BY_STEP_WALKTHROUGH.md) — command-by-command demonstration.
- [Testing contract](TESTING_CONTRACT.md) — input and numerical rules.
- [Test matrix](TEST_MATRIX.md) — coverage and recorded results.
- [Environment](ENVIRONMENT.md) — supported hardware and software.
- [NCU analysis](ncu/rtx4060/27-profile-gui-analysis/README.md) — 27 profiles and per-configuration reports.
- [NCU manual-inspection status](ncu/rtx4060/27-profile-gui-analysis/MANUAL_INSPECTION_STATUS.md) — completed review boundary and outstanding image-transcription work.
- [Visible disassembly inventory](ncu/rtx4060/27-profile-gui-analysis/DISASSEMBLY_INVENTORY.md) — per-profile mnemonic index from the exported PTX/SASS text.
- [Execution evidence](EXECUTION_EVIDENCE_INDEX.md) — workflow runs and artifacts.
- [Experiment report history](EXPERIMENT_REPORT.md) and [full report draft](../Report.md) — previous write-ups, not the final report.

Current evidence boundary (5 November 2026): native RTX 4060 validation run [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189) completed successfully, including the genuine cuTile smoke, 57 required GPU correctness/edge executions, optional attention-objective tests, the registered numerical matrix, CLI/NumPy verification, benchmark sessions and profiler stages. CPU/reference run [#42082132110](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42082132110) passed the CPU/reference suite and Python syntax checks. The mentor walkthrough has since been corrected to collect test counts dynamically and to run from the exact commit on the self-hosted runner. Those earlier runs are historical evidence only; the new walkthrough revision is not verified until its new Actions run completes.
