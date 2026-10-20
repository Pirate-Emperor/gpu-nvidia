# Execution and Artifact Evidence Index

This is the execution record for native runs, artifacts, runner diagnostics and their limits. It supplements rather than replaces `Report.md`, raw measurements and the implementation.

## Core native RTX 4060 validation

- Historical full native validation: [run #39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223) and [artifact #12149980979](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223/artifacts/12149980979).
- Demonstrated there: genuine cuTile smoke; 48 required GPU correctness tests; 29 legal case/tile combinations with zero mixed-tolerance violations; CLI versus independent NumPy oracle; three controlled benchmark sessions.
- Environment: Windows x64, GeForce RTX 4060 / SM89, driver 616.92, Python 3.12.10, CuPy 14.2.0, cuda-tile 1.6.0.
- Historical performance headline: longer batched D=128, tile 64×128, ~0.919 ms median / ~0.925 ms P90 / 18.62× against an intentionally unfused FP32 CuPy baseline. It is not a comparison with FlashAttention or vendor-optimized SDPA.

## Fresh native results — 20 October 2026

### Actual cuTile Nsight Compute profiles

- Latest completed dedicated profile workflow: [run #39414502490](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490), artifact [#12261445684](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39414502490/artifacts/12261445684), 1,074,786 bytes; zip SHA-256 5a544007115bbe5fa1d97b4d27699fe1c49bc1f83964d33f616a18e80eddeaae.
- The artifact contains eight native .ncu-repz reports, matching Details/Raw CSV exports and **seven verified genuine 1936×1048 Nsight Compute application-window screenshots**. The skipped screenshot was the longer-batched LOAD_LATENCY=4 report because Windows GetWindowRect failed; the capture manifest records the reason.
- The latest completed full native core workflow [#39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438) also captured fresh default-hint reports for medium D=128 / 64×64 and longer-batched D=128 / 64×128 in [artifact #12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195).
- A three-session timed experiment then found a credible LOAD_LATENCY=10 candidate for medium D=128 / 64×64. The dedicated profiler workflow now includes a matched medium hint-10 report; run [#39423368441](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423368441) was queued at this evidence snapshot. No hint-10 NCU resource values are claimed yet.

### Full native regression and measured hint sweep

- Latest completed full native Windows validation: [run #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438), commit bf5a6233e195c2ced647b4e20b5b2d914da60e85; artifact [#12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195), 194,711 bytes; zip SHA-256 e3eaf4ae1e88d2547d2034dfe333db86b70f69e2f2d6c71b0d67b7c8bdde0c32.
- Passed: genuine cuTile smoke; 48/48 required core GPU correctness tests; 3/3 optional GQA/MQA and packed-variable-length GPU tests; all 29 independent numerical case/tile combinations with zero tolerance violations; CLI output against the independent NumPy oracle; three controlled benchmark sessions; all accuracy-gated latency-hint trials; and two actual cuTile NCU captures.
- Benchmark controls: ten warmups, 30 CUDA-event samples, five launches per sample, three sessions with rotated/reversed tile order. Fresh summary:

| Workload | Tile | Median (ms) | Speedup vs unfused FP32 CuPy | Across-session range |
|---|---:|---:|---:|---:|
| Long D=128 | 64×64 | 0.267059 | 16.49× | 0.9% |
| Longer batched D=128 | 64×128 | 0.915558 | 18.69× | 0.5% |
| Medium D=128 | 64×64 | 0.058368 | 6.70× | 9.8% |

- The three-session latency-hint sweep passed all 36 accuracy-before-timing trials. Medium D=128 / 64×64 with LOAD_LATENCY=10 measured 0.053248 ms median versus 0.058342 ms default (8.73% gain); candidate and default session ranges were both about 0.53%, clearing the configured 1.00% noise-aware threshold. Longer-batched D=128 / 64×128 did not produce a credible gain: its best candidate (hint 2) gained only 0.16%, below a 5.88% threshold under that run's session spread. Keep LOAD_LATENCY=0 as the global default until the matched medium hint-10 NCU report is inspected.
- These are native Windows RTX 4060 cuTile results versus the intentionally unfused FP32 CuPy baseline, **not** comparisons against FlashAttention or vendor-optimized PyTorch SDPA. PyTorch integration is a separate gate.

### PyTorch SDPA, core-regression and optional-objective workflow

- Native SDPA run [#39422402415](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402415), after the flattened-batched torch.bmm backward change, failed all four batch/head autograd cases with CUBLAS_STATUS_INVALID_VALUE. Artifact: [#12265041153](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402415/artifacts/12265041153).
- A new diagnostic test reproduced CUBLAS_STATUS_INVALID_VALUE in plain float32 torch.bmm **before invoking the cuTile bridge**: [run #39423190840](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840), artifact [#12263887676](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840/artifacts/12263887676). This points to a PyTorch/cuBLAS wheel/runtime issue in the installed environment, rather than a cuTile-kernel-only bug. The subsequent FP16 model-block CUBLAS_STATUS_NOT_INITIALIZED followed those GEMM failures and is not independently diagnostic.
- To test whether the behavior is tied to the CUDA 13.0 PyTorch wheel, [scripts/torch_cublas_diagnostic.py](../scripts/torch_cublas_diagnostic.py) runs 2D GEMM, batch-1 and batch-2 BMM, and FP16 linear in isolated child processes. Run [#39423804493](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423804493) creates a separate project-local CUDA 13.2 PyTorch 2.14.0 virtual environment without overwriting .venv-cutile-windows; it was queued at this snapshot. No result for the alternative wheel, no PyTorch autograd pass, and no SDPA/vendor timing are claimed yet.
- The core remains independently validated by run #39422402438. Do not conflate the PyTorch failure with failure of the actual cuTile core, GQA/MQA, packed-varlen, numeric, CLI, or timing gates.
- Next gate: establish plain PyTorch GEMM/linear health; then pass all autograd and model tests; only afterwards capture actual SDPA backend/operator identity, output accuracy, and repeated timing JSON/CSV.

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


## Current source changes and queued native follow-ups

- The custom autograd bridge passes detached views to CuPy's DLPack importer while saving the original tensors for backward. Its flattened torch.bmm reference backward remains unvalidated because plain GEMM in the installed CUDA 13.0 wheel failed before the bridge was called.
- The extra test in commit [#92e6a04](https://github.com/xemptynothing/gpu-nvidia/commit/92e6a04f300fbe9c059c8b34a0f30ab59230594e) isolates plain PyTorch BMM health; run [#39423190840](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840) failed this sanity check with CUBLAS_STATUS_INVALID_VALUE.
- The isolated CUDA 13.2 wheel probe is tracked by [run #39423804493](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423804493) and was queued at this snapshot. Do not change the primary venv until its results establish whether basic GEMM and linear work in the alternate build.
- The NCU workflow was extended by [commit #d9837db](https://github.com/xemptynothing/gpu-nvidia/commit/d9837db33c416f63547d9f376ba8a583996b8c0f) to add the medium D=128 / 64×64 LOAD_LATENCY=10 capture plus Details/Raw CSV exports. Its run [#39423368441](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423368441) was queued at this snapshot.
