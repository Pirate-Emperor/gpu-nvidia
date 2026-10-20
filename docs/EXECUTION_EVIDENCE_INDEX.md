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

### Full native regression and measured hint sweep

- Workflow [#39415104135](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39415104135) succeeded on commit `90dafef5318a19e8752a7936643319eb8713083f`; [artifact #12261182182](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39415104135/artifacts/12261182182), 194,346 bytes, SHA-256 `06247a83344f9ef07ec8f69fa732c655c4b4bc99882d72035e0ab33adf449eda`.
- Passed: genuine cuTile smoke; 48/48 required GPU correctness tests; 3/3 optional GQA/MQA/packed-varlen correctness tests; all 29 numerical case/tile combinations; actual cuTile CLI output against the independent NumPy oracle; three controlled benchmark sessions; and the load-latency hint sweep.
- Benchmark controls: ten warmups, 30 CUDA-event samples, five launches per event sample, three sessions with rotated/reversed tile order.
- Fresh benchmark summary:

| Workload | Tile | Median (ms) | P90 (ms) | Speedup vs unfused FP32 CuPy | Across-session range |
|---|---:|---:|---:|---:|---:|
| Long D=128 | 64×64 | 0.268083 | 0.269128 | 16.45× | 0.50% |
| Longer batched D=128 | 64×128 | 0.915046 | 0.920576 | 18.70× | 0.59% |
| Medium D=128 | 64×64 | 0.055501 | 0.056955 | 7.02× | 8.86% |

- The latency-hint report used the same accuracy-before-timing gate across all 36 candidate trials; every trial passed. For medium D=128 / 64×64, hint 10 gained 8.27% versus the sweep's default median, but the noise-aware required gain was 9.88%. For longer-batched D=128 / 64×128, the strongest candidate (hint 6) gained only 0.10% versus a 1.00% minimum threshold. **Neither recommendation is credible; keep LOAD_LATENCY=0 for both workloads.**
- The general native-validation run covers core and selected optional cuTile objectives; it does **not** run the separate PyTorch autograd integration tests. Those remain tracked below.

### PyTorch SDPA, core-regression and optional-objective workflow

- First native run: [#39403488341](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403488341), failed during test collection due to a missing module-scope `torch` import; partial artifact [#12255372678](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39403488341/artifacts/12255372678). Follow-up: [#39414502429](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502429), passed collection and core gates but failed two autograd bridge tests on `BufferError: Can't export tensors that require gradient, use tensor.detach()` and then logged `CUBLAS_STATUS_NOT_INITIALIZED` during the model-block test. The bridge now detaches only the DLPack views; another native run must verify this fix and resolve any remaining cuBLAS issue.
- Passed before the collection error: PyTorch 2.14.0+cu130 was present in the project-local `.venv-cutile-windows`; PyTorch reported CUDA build 13.0 and `torch.cuda.is_available() == True`; CuPy/cuTile/PyTorch import check; cuTile smoke; the required core GPU correctness suite; 29-case/tile accuracy matrix; actual-cuTile CLI output and independent NumPy-oracle verification.
- Current targeted fix: the reference backward now flattens batch and head into an explicit GEMM batch and makes transposed operands contiguous before `torch.bmm`, avoiding the non-contiguous singleton-head GEMM path implicated by the trace. This change is not yet validated until the next native PyTorch run finishes. The previous SDPA run stopped before writing its timing JSON/CSV; no vendor comparison, integration timing, or gradient pass is claimed from it.
- Next gate: inspect the new native PyTorch run after the contiguous-batched-GEMM fix. Require all autograd and model tests to pass first, then verify SDPA operator identity, NumPy/cuTile output accuracy, and its three-session JSON/CSV before recording any framework speedup.

## Latest native profiler capture and genuine GUI screenshots

- Workflow: [run #39414502490](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490), successful on commit `2fd70a36248650c778b2353ef45e7762cb568c22`.
- Artifact: [#12261445684 — cutile-rtx4060-ncu-profiles](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490/artifacts/12261445684), 1,074,786 bytes; zip SHA-256 `5a544007115bbe5fa1d97b4d27699fe1c49bc1f83964d33f616a18e80eddeaae`.
- Verified artifact contents: eight native `.ncu-repz` reports, Details and Raw CSV exports, capture logs/manifest, and **seven actual Nsight Compute application-window PNGs**, each 1936×1048, checked after download. The screenshot shows NCU's own window with the actual report and metrics, not a synthesized visualization.
- Captured files: `cutile-profile-medium.png`, `cutile-profile-medium-stalls.png`, `cutile-profile-medium-latency4.png`, `cutile-profile-medium-latency8.png`, `cutile-profile-long-batched.png`, `cutile-profile-long-batched-stalls.png`, and `cutile-profile-long-batched-latency8.png`.
- One capture remains incomplete: `cutile-profile-long-batched-latency4.ncu-repz` was skipped because Windows `GetWindowRect` failed for that window. No screenshot is claimed for it. The seven successful PNGs plus all eight reports are in the artifact above.
- Latest profiler-only base observations: medium D=128 / 64×64 = 53.31 µs, 231 registers/thread, 65.54 KiB static shared memory/CTA, 8.34% achieved occupancy; longer-batched D=128 / 64×128 = 989.98 µs, 255 registers/thread, 81.92 KiB static shared memory/CTA, 8.31% achieved occupancy. These multi-pass NCU durations are not the ordinary CUDA-event benchmark.
- Scheduler exports: no eligible warp for 85.94% (medium) and 85.25% (longer-batched) of scheduler cycles, 0.14/0.15 eligible warps per scheduler on average, and 7.08/6.80 warp cycles per issued instruction. NCU speedup/issue-stall figures remain heuristics, not measured optimization gains.
- Hint 4/8 exports are present for both workload shapes. Because they are captured under the profiler and not the required repeated timing sweep, they do not promote a hint; keep LOAD_LATENCY=0 pending timed/accuracy evidence.

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


## Latest source fix and pending reruns

- Commit [#90dafef](https://github.com/xemptynothing/gpu-nvidia/commit/90dafef5318a19e8752a7936643319eb8713083f) updates `src/torch_integration.py`: DLPack now receives detached views of Q/K/V while the custom autograd context stores original tensors for backward.
- Full native-validation job [#39414730179](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414730179) passed setup and core/optional correctness gates and was running the three-session CUDA-event benchmark at the last check. Its final artifact must be inspected before benchmark results are asserted.
- Current-source native SDPA rerun [#39415104113](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39415104113) and full native regression [#39415104135](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39415104135) are queued on commit `90dafef5318a19e8752a7936643319eb8713083f`. They must finish before declaring PyTorch integration, SDPA comparison, autotuning, or load-latency timed sweep complete.
