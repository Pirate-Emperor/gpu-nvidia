# Run and Automation Evidence — separate from Report.md

This file intentionally contains workflow links, run-specific provenance, and GitHub/runner-only failures that are excluded from the technical report.

## Validated core runs

- Full native RTX 4060 validation: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223
- Dedicated actual-cuTile Nsight Compute capture/export: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066
- Four-profile artifact: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039
- Full test/benchmark/status artifact: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979
- Most recent completed CPU/reference CI before the long-running optional experiment: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39239915103 — 764 passed, 68 skipped, 0 failed. GPU-specific tests are skipped on CPU-only CI when CuPy/CUDA is absent; it is not a GPU test pass.

## Runner / GitHub failures intentionally excluded from Report.md

- Native validation rerun #39234069849 initially failed while resolving GitHub action-download metadata. The retry passed setup but failed on checkout with Windows Schannel error SEC_E_MESSAGE_ALTERED before dependencies, smoke, correctness, benchmark or profiler steps ran. This is a GitHub/runner transport failure, not a kernel/CUDA result. It does not negate the earlier successful native cuTile run.
- The first no-change PyTorch preflight used a PowerShell script step; the host's execution policy refused to load the generated PowerShell script. Python/pip commands never ran and no package changes occurred. The workflow was changed to use the Windows command shell without changing or bypassing the execution policy.
- The second PyTorch preflight completed successfully with pip --dry-run only. It resolved the official Windows/Python 3.12 CUDA 13.0 wheel and its declared support dependencies; it did not install PyTorch.

## Current optional benchmark state

- PyTorch dry-run preflight: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39238680934 — success, no installation.
- Actual PyTorch SDPA install/benchmark workflow: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39239914946 — last observed queued; no test values or installation result had been produced at this snapshot.
- Native GPU re-validation after adding the SDPA script: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39239855079 — last observed queued; this should be checked before deciding whether a fresh run reached any project steps.

## Screenshot provenance

The SVGs in docs/assets/report are derived from the exported NCU metrics; they are charts, not application screenshots. Four native profiler reports and their Details/Raw CSV exports have been preserved. Genuine Nsight Compute GUI screenshots or a desktop clip have not yet been captured; they must not be represented as done until the images/media files are generated and inspected.

## Latest optional-objective implementation and validation boundary

- Added isolated optional APIs, references and tests for GQA/MQA, packed variable-length sequences, PyTorch autograd/model integration, correctness-gated tile autotuning, and a cuTile DRAM-load-latency hint sweep. CPU/reference CI validates pure references, recommendation policy and script syntax; it does not validate the added GPU kernels.
- Current NCU evidence workflow includes hint=4 and hint=8 basic captures for the medium D=128 (64×64) and longer-batched D=128 (64×128) workloads. It also invokes scripts/capture_ncu_screenshots.py for real application-window snapshots; the screenshot gate is **pending a successful capture manifest and PNG inspection**.
- The latency-hint sweep tests 0/2/4/6/8/10 with three rotated sessions, correctness-before-timing, 10 warmups, 30 CUDA-event samples, and five launches averaged per sample. No hint is promoted unless the median advantage exceeds a noise-aware threshold.
- The current native GPU run must validate the compile-time KV_GROUPS=1 and LOAD_LATENCY=0 default parameters as well as the GQA/MQA and packed-varlen optional gates. A previously successful core baseline does not prove these later source changes are passing.
- The approved PyTorch job now includes: local-venv-only CUDA wheel installation, SDPA backend/numerical comparison, autograd/model-block test, end-to-end model-block latency, optional GPU correctness tests and autotuning. It is not complete until installation succeeds and the output artifacts are inspected.
- A workflow-edit mistake temporarily produced an invalid screenshot step; it was restored and rewritten as a separate Python helper before the current screenshot workflow was queued. This is kept here because it concerns automation configuration, not project/CUDA behavior.

