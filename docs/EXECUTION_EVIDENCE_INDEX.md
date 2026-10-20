# Execution and Artifact Evidence Index

This is the execution record for native runs, artifacts, runner diagnostics and their limits. It supplements rather than replaces `Report.md`, raw measurements and the implementation.

## Core native RTX 4060 validation

- Historical full native validation: [run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223) and [artifact #12149980979](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979).
- Demonstrated there: genuine cuTile smoke; 48 required GPU correctness tests; 29 legal case/tile combinations with zero mixed-tolerance violations; CLI versus independent NumPy oracle; three controlled benchmark sessions.
- Environment: Windows x64, GeForce RTX 4060 / SM89, driver 616.92, Python 3.12.10, CuPy 14.2.0, cuda-tile 1.6.0.
- Historical performance headline: longer batched D=128, tile 64×128, ~0.919 ms median / ~0.925 ms P90 / 18.62× against an intentionally unfused FP32 CuPy baseline. It is not a comparison with FlashAttention or vendor-optimized SDPA.

## Fresh native results — 20 October 2026

### Actual cuTile Nsight Compute profiles

- Workflow: [run #39403495373](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403495373), completed successfully on commit `affd935649c2cf9967210769b0381d807c6c4d6a`.
- Artifact: [#12256589808 — cutile-rtx4060-ncu-profiles](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403495373/artifacts/12256589808), 631,797 bytes; SHA-256 reported by Actions: `d8ddf993d41b4bccc4aaf1d1b50c2c917c93c02a347e6d51362521e562d59fef`.
- The artifact includes eight genuine `.ncu-repz` reports with Details/Raw CSV exports and logs: baseline profiles and scheduler/warp-state profiles for medium D=128 64×64 and longer-batched D=128 64×128, plus LOAD_LATENCY=4 and =8 profiles for both cases.
- Latest baseline profile durations were 52.86 µs (medium) and 987.94 µs (longer-batched). Latest scheduler snapshots showed no eligible warp for 85.90% and 85.27% of scheduler cycles, respectively; the reports show about 8.3% achieved occupancy due chiefly to shared-memory residency. These are profiler measurements/diagnostics, not replacements for ordinary CUDA-event benchmark medians.
- Hint 4/8 captures do **not** settle which hint is faster. The native multi-session CUDA-event timing/accuracy sweep has not yet finished.
- Screenshot status: **0 PNGs captured**. The artifact's capture manifest explicitly reports no screenshots. No GUI screenshot is claimed; only the native report files and CSV/Details exports are available from this run.

### PyTorch SDPA, core-regression and optional-objective workflow

- First native run: [#39403488341](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403488341), failed during test collection due to a missing module-scope `torch` import; partial artifact [#12255372678](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403488341/artifacts/12255372678). Follow-up: [#39414502429](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502429), passed collection and core gates but failed two autograd bridge tests on `BufferError: Can't export tensors that require gradient, use tensor.detach()` and then logged `CUBLAS_STATUS_NOT_INITIALIZED` during the model-block test. The bridge now detaches only the DLPack views; another native run must verify this fix and resolve any remaining cuBLAS issue.
- Passed before the collection error: PyTorch 2.14.0+cu130 was present in the project-local `.venv-cutile-windows`; PyTorch reported CUDA build 13.0 and `torch.cuda.is_available() == True`; CuPy/cuTile/PyTorch import check; cuTile smoke; the required core GPU correctness suite; 29-case/tile accuracy matrix; actual-cuTile CLI output and independent NumPy-oracle verification.
- Root causes so far: (1) `_CuTileCausalAttention(torch.autograd.Function)` required the module-scope `torch` import, now present; (2) DLPack input conversion rejected Q/K/V tensors that require gradients, now fixed with `q.detach()`, `k.detach()`, and `v.detach()` for the CuPy views while original tensors remain saved for backward. The cuBLAS initialization error is not yet explained. No SDPA timing, model-integration timing, optional-objective suite, autotuner recommendation, or load-latency timing result is claimed; the failed run stopped before these stages.
- Next gate: run the complete native workflow again on the fixed commit, inspect its JSON/CSV, verify SDPA operator identity and numerical accuracy, then update the report with measured results.

## Native Nsight Compute evidence from the earlier baseline

- Historical profiler workflow: [run #39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066), [artifact #12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039).
- The figures committed under `docs/assets/report/` are exported-metric plots, not screenshots of the Nsight Compute application.

## CPU/reference CI and limits

- CPU/reference suite: [run #39403495477](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403495477) passed after the PowerShell/source-fetch fix. CPU-only CI does not prove GPU compilation, execution, numerical agreement or performance.
- Native GPU workflows use the Windows self-hosted RTX 4060 runner; the Linux A100/ordinary-CUDA control environment is separate and its results are not used as proof of native RTX 4060 cuTile execution.

## Historical runner failures

- [Run #39234069849](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39234069849) failed before GPU tests: first action download metadata resolution failed; its retry later hit a Git/TLS `SEC_E_MESSAGE_ALTERED (0x8009030F)` transport error during checkout.
- Earlier native jobs also failed before testing when Windows PowerShell refused the generated script under its execution policy. Native PowerShell steps now use `powershell -NoProfile -ExecutionPolicy Bypass -File {0}`.
- These are runner/transport failures, not evidence of a CUDA kernel defect.

## Source-of-truth and report locations

- Canonical technical report: [Report.md](../Report.md).
- Project two-pager/product scope: [PROJECT_TWO_PAGER.md](PROJECT_TWO_PAGER.md).
- Completion checklist: [COURSE_PROJECT_COMPLETION_CHECKLIST.md](COURSE_PROJECT_COMPLETION_CHECKLIST.md).
- Profile interpretation: [CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md).
- Detailed benchmark snapshot: [CUTILE_RTX4060_RESULTS_2026-10-19.md](CUTILE_RTX4060_RESULTS_2026-10-19.md).


## General native-validation shell fix — 20 October 2026

- [Run #39414502390](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502390) failed in its first dependency step before installing packages or running any GPU checks. The job log shows the runner's default shell attempted to dot-source the generated temporary PowerShell script and received `PSSecurityException: running scripts is disabled on this system`.
- The workflow `.github/workflows/cutile-rtx4060-docker.yml` has been updated so each PowerShell step explicitly invokes `powershell -NoProfile -ExecutionPolicy Bypass -File {0}`, matching the already-working native NCU/SDPA workflow pattern. The rerun result on the fix commit is pending; the failed run is not a kernel result.
