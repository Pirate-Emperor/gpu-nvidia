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
