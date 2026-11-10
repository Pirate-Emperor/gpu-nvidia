# Project Progress Log

This file is the durable experiment log for the project. Record what was tested, what passed, what failed, and what remains blocked.

## Current status note — 24 October 2026

The last complete RTX 4060 run is #39422402438 on commit bf5a623. It passed the 48 core GPU tests, 29 numerical configurations, CLI check, three timing sessions and two Nsight Compute configurations. The kernel was compacted after that run, so current main still needs a fresh native GPU pass. Treat the dated entries below as history, and check the [current README](../README.md) and [evidence index](EXECUTION_EVIDENCE_INDEX.md) before quoting a result.

## 2026-10-05 / 2026-10-06 — Foundation

### Repository foundation
- Added an independent FP32 NumPy mathematical reference.
- Added a tiled online-softmax NumPy reference.
- Added a deliberately unfused CuPy GPU baseline.
- Added a scalar CUDA baseline.
- Added an optional PyTorch SDPA reference.
- Added the first cuTile fused causal-attention kernel.
- Added required launch-contract validation.
- Added deterministic input generation and exact NPZ/NPY I/O.
- Added correctness/contract/I/O/generator/online-reference tests.
- Added a frozen representative benchmark matrix.
- Added a persistent human-readable test/validation ledger at docs/TEST_MATRIX.md.

### Expanded test coverage
- Added randomized finite-output coverage across B/H/N and D=32/64/128.
- Added exact prefix-mean checks for zero logits.
- Added constant-V invariance checks across all required head dimensions.
- Added extreme finite-logit stability checks.
- Added convex-hull invariants for attention outputs.
- Added explicit batch/head isolation checks.
- Added additional current-query locality checks.
- Added GPU-only tests for asymmetric legal tile configurations, tile-to-tile agreement, causal future-K/V invariance, constant-V behavior, zero-logit prefix means, nonfinite input rejection, non-contiguous rejection, and unsupported/mismatched dimensions.

### Independent native CUDA work
- Added a custom block-row fused CUDA kernel that uses ordinary CUDA C++ and cooperative reductions.
- Added a second warp-row fused CUDA kernel using warp-shuffle reductions.
- Added a third shared-memory tiled fused CUDA kernel with configurable 32/64 key tiles.
- Added independent correctness suites against the FP32 oracle and tiled online-softmax reference.
- Added cross-implementation tests between the two custom fused CUDA variants.
- Added a native-CUDA GitHub Actions workflow that deliberately does not require cuTile or R580+, so the R550 node can validate these implementations independently.
- Added a baseline benchmark path comparing unfused CuPy, scalar CUDA, block-row fused CUDA, warp-row fused CUDA, and shared-memory tiled CUDA.

### CPU validation
- The self-hosted A100 runner successfully executed the CPU/reference suite.
- The corrected suite reached 75 passing tests before this latest edge-case expansion.
- The newly added CPU tests are preserved and will run automatically through the repository test workflow.
- One initial test failure was a test-shape assertion, not an attention implementation failure; it was fixed and re-run successfully.

### A100 environment
- The self-hosted runner selects an NVIDIA A100-SXM4-40GB node.
- The node reports 40 GiB and compute capability 8.0.
- Current driver is R550.90.07.
- Current NVIDIA cuTile Python releases require a newer driver, so the production cuTile workflow stops at an explicit environment gate.
- A read-only access audit reports UID/EUID 55012, not root; sudo exists but passwordless sudo is unavailable and noninteractive sudo policy inspection requires credentials.
- Conclusion: a host-driver upgrade cannot be performed by the GitHub runner account itself. It requires an administrator/privileged maintenance path, unless the cluster provides a separately authorized upgrade mechanism.

### Forward-compatibility experiment
- NVIDIA cuda-compat-13-4 was downloaded from the NVIDIA repository without root.
- The compatibility libraries were extracted into the job workspace.
- Native CUDA 13.x access failed with cudaErrorInsufficientDriver.
- Loading the extracted compatibility libraries produced cudaErrorSystemDriverMismatch (CUDA error 803).
- Therefore the current R550 node has not been validated as a cuTile runtime through that compatibility configuration.

### cuTile 1.2 experiment
- A branch attempted to test an early cuTile release associated with Ampere/SM80 support.
- The currently reachable Python package index does not expose cuda-tile 1.2.0, so that branch did not reach GPU execution.

## Engineering infrastructure added
- Added a project-structure test that freezes required deliverables.
- Added a read-only driver-access audit workflow.
- Added benchmark and Nsight workflows that are ready for a supported A100/cuTile runtime.
- Added result-storage conventions and a persistent status snapshot.

## Baseline ladder
1. NumPy FP32 full attention.
2. NumPy tiled online-softmax.
3. Unfused CuPy attention.
4. Simple scalar CUDA attention.
5. Optional PyTorch SDPA reference.
6. Fused cuTile attention.

Each executable GPU backend has an independent correctness-test path against the mathematical oracle.

## Next execution gate
1. Ask the cluster administrator for an R580+ driver path or another supported A100 node.
2. Run the expanded GPU correctness matrix.
3. Capture at least two Nsight Compute profiles.
4. Run >=30-iteration benchmark sessions and preserve raw JSON/CSV.
5. Optimize only after correctness and profiling evidence exist.

## Experiment policy
For each material kernel change:
1. preserve/update the relevant tests;
2. record the exact environment and configuration;
3. record successful and unsuccessful outcomes;
4. re-run correctness before claiming a performance improvement;
5. preserve benchmark/profiler artifacts instead of replacing prior runs.


## 2026-10-06 — Current development direction

### Driver-independent engineering
The project is no longer blocked on cuTile for algorithmic validation. The custom
CUDA variants are intended to exercise the same fused mathematical path:
scaled QK scores -> causal restriction -> stable max-shifted softmax -> V
accumulation, with FP16 inputs and FP32 accumulation.

The native CUDA workflow is intentionally separate from the cuTile workflow.
A successful native-CUDA run would validate the ordinary CUDA implementations
on the existing A100/R550 node without implying that cuTile is supported.

### Local execution note
The development sandbox cannot resolve github.com, so the latest repository
state was not executed locally from this environment. CI remains the source of
truth for execution results. No unexecuted test or benchmark is marked PASS.

## 2026-10-06 — CI sanity finding and correction

The first CI execution after the expanded test additions completed 94 tests and
reported 5 failures plus 11 GPU/package skips. The failures were all in newly
added CPU/reference tests, not GPU execution:
- constant-value exact-equality assertions were too strict for FP32 reduction;
- one extreme-logit expected array had a missing leading B/H dimension;
- the common-key-shift metamorphic property used an overly strict tolerance
  despite FP16 input rounding.

These test specifications were corrected on main. The GPU audit run remains a
separate self-hosted A100 job and must not be inferred from this hosted CPU run.

## 2026-10-06 — Driver maintenance is deferred; development continues

The cluster administrator has confirmed that an R580+ driver upgrade is possible during the next scheduled maintenance cycle, but the cycle may be delayed and no other available node currently provides R580+.

Therefore the project intentionally continues without cuTile execution rather than waiting:
1. validate independent CUDA implementations on the existing A100/R550 host;
2. benchmark unfused and fused ordinary-CUDA variants;
3. validate both A100 devices exposed to the runner;
4. harden correctness, edge-case, I/O, and reproducibility tests;
5. collect native CUDA/Nsight evidence where the current software stack permits;
6. after the driver upgrade, install/validate cuTile and run the same frozen correctness/performance matrix.

Important: native CUDA results are development/control results. They are not presented as cuTile results and do not replace the required cuTile implementation.


## 2026-10-08 / 2026-10-09 — Regression freeze and optimization experiments

The consolidated evidence and performance tables live in [docs/EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md); use that report rather than reconstructing results from this chronological log.

### Confirmed correctness state
- A100 native CUDA run #44 on `a8172749`: **133 passed, 1 skipped**. It followed the fix to an invalid affine-property test that incorrectly requested unsupported `tile_k=2`; the kernel's supported tile sizes remain 16/32/64.
- Broad non-cuTile run #47 on the corrected test commit: **905 passed, 52 skipped**. Latest rerun #51: **905 passed, 52 skipped, 0 failed**.
- Latest CPU/reference run #285: **761 passed, 68 skipped, 0 failed**. Hosted CPU CI skips GPU tests when CuPy/GPU dependencies are not present.
- Dedicated v4 and v5 GPU correctness workflows: **30/30 passed each**. These variants remain experiments, not promoted defaults.

### Version ladder and compiler evidence
- Added `benchmarks/benchmark_tile_ladder.py`: v1/v2/v3/current comparisons across frozen cases and tile_k 16/32/64, with 10 warmups, 30 CUDA-event iterations, median/P90, and first-compile time measured separately.
- Ladder run #3 measured first compile as v1 1652 ms, v2 717 ms, v3 1063 ms, current 808 ms. Per-case kernel medians and limitations are documented in the experiment report.
- Added `scripts/ptxas_tile_report.py` and ran it on SM80. v1/v2/v3/current/v4 compiled without reported spills. v3 uses 40 registers; current and v4 use 32 registers and emitted identical reported shared-memory usage/cubin size.
- v4 tries persistent Q registers across K tiles. It passed correctness but had mixed timing, regressed a long-batched case, and generated code equivalent to current; it is not promoted.
- v5 tries `half2` QK loads. It passed 30 correctness tests; selected cases were faster, but per-run/per-tile timing variability and mixed regressions make the result inconclusive. Current remains the main A100 surrogate until a controlled repeat confirms a robust gain.

### Profiling and runtime blockers
- Native Nsight workflow detected Nsight Compute 2024.1.1 and both A100s, but profiling failed with `ERR_NVGPUCTRPERM`; no kernels/counters were profiled.
- CUDA 13.4 forward compatibility was not functional on R550 (insufficient-driver error / CUDA error 803). The CUDA 13.0 package/runtime attempt did not establish a verified working cuTile GPU run. Current cuTile execution remains blocked by the R580+ runtime gate.
- The ordinary CUDA A100 work is explicitly a cuTile-like surrogate, not actual NVIDIA cuTile.

### Documentation
- Added [docs/README.md](README.md) as the documentation map.
- Added [docs/EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the consolidated test ledger, all approach families, timing results, PTXAS evidence, infrastructure attempts, and next decisions.


## 2026-10-10 — Repeated v5 benchmark and full PTXAS comparison

### Compiler resource check
- Added `src/variants/shared_tile_v5.py` to `scripts/ptxas_tile_report.py`, enabled the workflow to archive the JSON output, and ran [PTXAS workflow #5](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38054963187) successfully.
- SM80 PTXAS reports zero spill loads/stores for v1/v2/v3/current/v4/v5. v5 uses 32 registers and 33,792 B shared memory, equal to current; v5's cubin is 15,520 B vs current's 15,136 B.

### Controlled v5 comparison
- Reworked `benchmarks/benchmark_v5.py` to compare current and v5 in paired sessions, alternating which version runs first. Each version/session uses 10 warmups and 30 CUDA-event samples; three repeats are recorded for every one of seven cases and each tile_k=16/32/64. Raw session JSON/CSV and a summary CSV are uploaded by [A100 v5 run #4](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001163).
- Correctness stage passed **30/30 tests** before timings. The run succeeded.
- Measured v5 gains were modest on medium D=64, mixed across short cases, and only about 0.3–0.6% on long/batched D=128. Ratios varied widely for some short cases. v5 remains unpromoted because the data do not establish a robust general win.
- Latest CPU/reference run [#306](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055001131) passed **761 tests, 68 skipped**.

### Remaining blockers
- Actual cuTile still needs a supported driver/runtime and an actual successful compile/launch/correctness run.
- Nsight Compute counter profiling remains blocked by `ERR_NVGPUCTRPERM`; the successful PTXAS report is compile-time evidence only, not hardware-counter profiling.


## 2026-10-10 — Nsight Systems timelines for the A100 CUDA surrogate

- Added `.github/workflows/a100-nsys-timeline.yml` to capture two warmed ordinary-CUDA shared-tile launches without requiring Nsight Compute hardware counters.
- [Workflow #1](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055492320) completed successfully on A100/R550. The runner/version check and both capture commands passed; artifact upload succeeded.
- Uploaded [artifact #11671195371](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055492320/artifacts/11671195371) containing two Nsight Systems reports: medium D=128 with tile_k=64, and longer batched D=128 with tile_k=32.
- This provides CUDA API/kernel timeline evidence for the ordinary-CUDA surrogate, not cuTile and not Nsight Compute hardware-counter metrics. Actual-cuTile profiling remains open.


## 2026-10-10 — Nsight Systems summary export and cuTile workflow readiness

- Improved the timeline workflow to run `nsys stats` for `cuda_gpu_kern_sum` and `cuda_api_sum` after capturing the warmed traces.
- [Nsight Systems run #2](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950) completed successfully. The artifact [#11671075954](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954) contains both timeline reports and summary outputs.
- Prepared `.github/workflows/a100-nsight.yml` to run `tests/test_gpu_correctness.py` and `tests/test_gpu_edge_cases.py` before profiling actual cuTile, and then profile two configurations (medium D=128, 64x64; longer batched D=128, 64x32).
- This actual-cuTile workflow is **prepared but not validated on hardware**; the current R550 node is blocked at its R580+ check. Nsight Compute counter access remains separately blocked by `ERR_NVGPUCTRPERM`.
- Updated `src/cutile_attention.py` to make the runtime-unavailable message hardware-agnostic; it no longer incorrectly directs all users to the A100 node.


## 2026-10-10 — Regression guardrails for documentation and workflow gates

- Added required-path checks for the TODO checklist, experiment report, current status snapshot, benchmark script, PTXAS script, and profiling/benchmark workflows.
- Added root-README navigation checks and a static workflow-contract test verifying that the future actual-cuTile profiling workflow runs correctness tests before two profile configurations.
- Updated `src/cutile_attention.py` so runtime import failures ask for a supported GPU/driver/package combination instead of incorrectly directing all users to the A100 node.
- CPU/reference run [#334](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264023): **763 passed, 68 skipped**.
- Full A100 non-cuTile run [#52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082): **907 passed, 52 skipped, 0 failed**.
- These checks validate project structure and the workflow's intended guardrails; they do not validate actual cuTile execution. That remains blocked by the current runner's R550 driver gate.


## 2026-10-18 — Handoff, benchmark ledger, and local runner security

No kernel source or benchmark algorithm was changed in this documentation-focused session. The goal was to make continuation in a new chat safe and reproducible.

- Re-verified that main before the handoff documentation update was afe530bc090c779ebe5d741dcf7ae544985570d1.
- Re-checked the broad A100 non-cuTile result: [run #52](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056264082), **907 passed / 52 skipped / 0 failed**. The workflow is ordinary CUDA/reference/project-contract coverage; it is not an actual-cuTile run.
- Re-checked the latest CPU/reference suite on the then-current main SHA: [run #340](https://github.com/SpryzenHell/gpu-nvidia/actions/runs/38056390931), **763 passed / 68 skipped / 0 failed**.
- Consolidated targeted v4/v5 correctness, the paired v5 benchmark, PTXAS resource results, Nsight Systems surrogate traces, and the blocked Nsight Compute/actual-cuTile gates into [the chat handoff](CHAT_HANDOFF_2026-10-18.md).
- Added a continuation prompt at [CONTINUE_PROMPT.md](CONTINUE_PROMPT.md).
- Documented local GPU inventory commands and a runner threat model: repository workflows execute on the runner; isolate by OS/container/VM, use an unprivileged ephemeral runner, mount only the project workspace, keep personal credentials out, and do not treat runner registration as a live ChatGPT shell.
- **Next decision:** obtain the user's local GPU/driver/toolkit output before choosing the exact Windows-native vs WSL2/Linux runner implementation. Do not request runner registration tokens or sudo credentials. Actual cuTile correctness and successful Nsight Compute profiles remain unverified.

## 2026-10-18 — Continuation audit and actual-cuTile routing decision

- Rechecked private repository `xemptynothing/gpu-nvidia`; `main` at the start of this session was `e87d8b9352fdf6783347d5c1e1945172011ebeb9`, matching the supplied handoff commit.
- Inspected the actual target `src/cutile_attention.py`, independent FP32 and tiled references, ordinary-CUDA variants v1–v5, GPU correctness/edge tests, benchmark/profile scripts, and project workflows. The target implementation exists, but no actual-cuTile compile/launch/correctness run was newly established in this continuation.
- **Routing decision:** R550.90.07 is below NVIDIA's documented R580+ cuTile Python prerequisite. Stop using the current A100 for actual cuTile; keep it for ordinary-CUDA surrogate tests. Do not repeat prior failed compatibility attempts without a materially different supported route. Actual cuTile should start on the user's inventoried local GPU or another explicitly authorized supported GPU environment; do not assume an RTX 4060.
- Verified NVIDIA's [cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html): Linux x86_64, Linux aarch64 and Windows x86_64; GPU compute capability 8.x–12.x; driver R580+; Python 3.10–3.14. Optional `[tileiras]` packages must match major.minor; detailed profiling requires R580.126.09+ on Linux or R582.16+ on Windows.
- **Actions access limitation:** **Actions evidence access note (2026-10-18 continuation):** the connected Actions run list returned zero visible runs; direct run-detail, job and artifact requests for handoff IDs (including #343, #52, #340, benchmark #4 and Nsight Systems #2) returned 404; the current handoff commit had no visible combined status. This does not prove historical runs are absent. Historical counts below remain from the prior handoff and were not independently re-audited through this connection. The previous handoff's pass/skip counts remain historical records, not counts independently re-audited in this continuation.
- Local GPU/OS/driver/NVCC inventory has not yet been supplied; therefore no Windows-native vs Linux/WSL2 decision, runner registration, claimed isolation boundary or actual GPU test occurred. This continuation makes documentation changes only.
- **Offline numerical sanity check (not GPU/cuTile execution):** compared the mathematical online-softmax recurrence with the independent FP32 full-matrix reference over N=128/256, D=32/64/128, and tiles 32x64, 64x32, 64x64 and 128x128. All 24 shape/configuration combinations remained finite; the maximum absolute error after FP16 output rounding was about 2.44e-4. Static analysis also confirms every query row has at least one valid key in key tile zero (key position 0 is causal for all query positions), so the initially suspected all-masked-first-tile `-inf - -inf` failure was not reproduced. A temporary speculative code/test edit was reverted; the committed kernel and edge-test source match their original content. This does not validate the compiled cuTile kernel.

## 2026-10-18 — RTX 4060 local inventory and Windows smoke harness

- Received the user's local inventory: NVIDIA GeForce RTX 4060, compute capability 8.9, driver 616.92, 8,188 MiB VRAM with 7,925 MiB free at capture, Windows 10 Pro x64 build 26300 / WDDM. `nvcc` and Python are unavailable; WSL2 and Docker are not installed.
- NVIDIA's current [cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) lists Windows x86_64, compute capability 8.x–12.x, driver R580+ and Python 3.10–3.14. The supplied GPU/driver/OS pass these published gates on paper; Python and a functioning cuTile/CuPy package stack still need to be installed and tested.
- Selected native Windows as the first bring-up path because it is a documented target OS and avoids introducing WSL2/Docker before the minimal smoke test. Driver 616.92 also exceeds the published Windows threshold (R582.16+) for detailed cuTile statistics, but Nsight Compute presence/access is unverified.
- Added `scripts/cutile_smoke.py` to compile, launch, synchronize and verify a real cuTile vector-add kernel; added `scripts/setup_cutile_windows.ps1` to install per-user/project-venv dependencies using `cuda-tile[tileiras]` and `cupy-cuda13x[ctk]` without changing the system driver/Toolkit; added `docs/LOCAL_GPU_WINDOWS.md` with the exact sequence and known Windows NVRTC caveat.
- Added `REQUIRE_CUTILE=1` test mode to `tests/test_gpu_correctness.py` and `tests/test_gpu_edge_cases.py` so missing CuPy/cuTile imports fail rather than silently skip for local acceptance runs. Registered the new files in the structural guard.
- This is build-up/tooling work only. The smoke script has **not** run on the user's computer; no claim of actual-cuTile compilation, attention correctness, benchmark, or profile is made. The next gate is an explicit `CUTILE_SMOKE=PASS` result, followed by targeted correctness tests.
- Do not register an Actions runner on the personal Windows host yet. First pass the manual smoke/correctness gates and then test an OS-enforced least-privilege boundary if Actions execution is still needed.


## 2026-10-18 — Local cuTile smoke passed; Docker runner prepared

### Actual local cuTile runtime
- Installed project-local CPython 3.12.10, CuPy 14.2.0, cuTile 1.6.0, and matching CUDA 13.4 TileIRAS/NVCC/NVVM Python packages on the Windows RTX 4060 (SM89, driver 616.92).
- Ran `scripts/cutile_smoke.py` locally. Evidence: `GPU_SYNCHRONIZE=PASS`, `OUTPUT_CHECK=PASS`, `CUTILE_SMOKE=PASS` for 4,096 vector-add elements.
- This proves a genuine cuTile kernel can compile/launch on the local GPU. It does **not** validate `src/cutile_attention.py`, causal masking, attention numerical correctness, performance, or profiling.

### Test collection fixes
- Fixed `tests/test_gpu_correctness.py` so pytest receives one explicit ID per complete parameter set instead of attempting to unpack an individual `AttentionCase`.
- Fixed the exploratory tile IDs in the same file to use one ID per tile pair.
- Fixed lazy runtime import scope in `src/cutile_attention.py`: `ct` is made visible in module globals before the cuTile decorator resolves `ct.Constant[int]` annotations.
- The user's prior attention test run stopped at collection before any attention kernel executed. The full GPU tests have not yet been rerun after these changes.

### Dockerized GitHub Actions runner
- Added a non-root Linux runner image, Docker Compose GPU configuration, ephemeral one-job entrypoint, and PowerShell launcher.
- Added `.github/workflows/cutile-rtx4060-docker.yml`, limited to manual dispatch on `main`, read-only contents permission, genuine smoke gate, and strict correctness/edge tests with `REQUIRE_CUTILE=1`.
- The launcher checks that Docker GPU passthrough sees the RTX 4060 before requesting a short-lived runner token, starts the runner, dispatches the workflow, watches the run, and tears down the container.
- Isolation controls: no host bind mounts, no Docker socket, no privileged mode, non-root user, dropped Linux capabilities, `no-new-privileges`, resource limits, ephemeral runner. Docker/WSL2 and host-kernel trust remain part of the security boundary.
- **Not yet executed:** the host inventory still has no WSL2/Docker Desktop. Install those prerequisites and GitHub CLI, authenticate locally with `gh`, then run `scripts/start_cutile_docker_runner.ps1`. No registration token or personal credential was written to the repository.

### Next gates
1. Launch the Docker workflow and resolve any container/build/runtime failures.
2. Require the actual causal-attention correctness matrix to pass; do not treat the smoke as attention validation.
3. Run the CLI/reference check and controlled benchmarks only after correctness passes.
4. Capture at least two successful actual-cuTile profiler runs and preserve raw artifacts.

## 2026-10-18 — Owner-supplied token and persistent runner repair

- Investigated the initial Docker runner failure from the observed console output. The image build and container start succeeded, but the script did not confirm that the runner reached GitHub's Online state. Its cleanup then failed because the launcher removed `RUNNER_NAME` before running Compose `down`; that final cleanup error did not establish the original registration failure's cause.
- Changed `scripts/start_cutile_docker_runner.ps1` to accept `-RunnerToken` supplied by the owner, avoid using GitHub CLI to mint tokens or dispatch workflows, check the probe image is already local rather than pulling it automatically, verify RTX 4060 passthrough, and start the runner detached.
- Changed `docker/cutile-runner/entrypoint.sh` to remove `--ephemeral`, so the runner remains registered across jobs until the owner deletes it in GitHub. Workflow dispatch is now a manual GitHub Actions UI step.
- Changed Compose token/name interpolation defaults so `docker compose down` works when those variables are no longer present in PowerShell. Updated `docs/CUTILE_DOCKER_RUNNER.md` with the startup, log-inspection and removal sequence.
- Updated the structural contract test to match the persistent owner-supplied-token mode. CPU/reference CI run [#75](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39147796099) passed **764 tests, 68 skipped, 0 failed**. The skips include GPU modules unavailable on GitHub's hosted CPU runner; this is not actual-cuTile validation.
- The user separately reported a local targeted GPU test run of 34 passed in 41.91 seconds; the copied console output did not establish the exact `REQUIRE_CUTILE` environment state, so treat this as user-reported test evidence, not as an independently audited Actions artifact.
- **Still unverified:** re-launch with the revised script, GitHub runner Online state, the main-only cuTile smoke/attention workflow, CLI/reference check, benchmarks and actual-cuTile profiles. Do not mark the kernel complete from smoke/CPU tests alone.


## 2026-10-19 — local Windows RTX 4060 runner bring-up and blocked cuTile execution

**Scope and constraints:** used the repository's self-hosted Windows runner at the checkout below the project directory. No sudo, administrator shell, package installation, external-tool installation, or dataset download was performed. Workflow source and experiment logs stayed in the repository's Actions workspace. The only installed tools invoked were GitHub's checkout/artifact workflow steps and the already-present nvidia-smi.

### Run log (UTC, exact workflow URLs and commits)

| Run | Commit | Result | What actually executed | Conclusion |
|---|---|---|---|---|
| [#39154733638](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154733638) | 87381e42ab5a25d2db8da3c155b66dce12dbedbd | **Failed: shell setup** | Runner setup and checkout succeeded. Step 3 failed with “pwsh: command not found”. All GPU steps were skipped. | Workflow incorrectly requested PowerShell Core on a host that has only Windows PowerShell. This is infrastructure, not kernel evidence. |
| [#39154791220](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154791220) | 1cfc16ddca9f0c7cd4506af0ce6a6dc94c55e04e | **Failed: environment path** | Switched workflow shells to built-in powershell. nvidia-smi detected **NVIDIA GeForce RTX 4060, driver 616.92, compute capability 8.9, 8188 MiB**. The configured .venv-cutile-windows Python did not exist at either candidate path. No package installation was attempted. | GPU runner visibility confirmed. Expected Python environment path was incorrect. Smoke, attention, CLI, benchmark and profile steps did not run. |
| [#39154888145](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154888145) | 2971f5ac0a9450ebaa2cfb5cfa3e52e279700676 | **Failed: missing in-repo environment** | Added repository-root search for pre-existing venvs and inspected Python Launcher registrations. No matching venv was found under the project root. The only discovered interpreter was the global CPython 3.12 at C:\Users\Raman Singh\AppData\Local\Programs\Python\Python312\python.exe; probing it failed because NumPy is not installed. | No dependency or tool was downloaded. GPU attention did not run. |
| [#39154979187](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979187) | b9189b5d8bec8666a0c22e1747639ba340e0d157 | **Failed: no permitted existing Python environment found** | Fixed PowerShell native-stderr handling so candidate import failures were captured and the diagnostic remained visible. RTX 4060 inventory passed again. The only environment candidate found under the project root was the global Python 3.12 interpreter; its probe exited 1 with “ModuleNotFoundError: No module named 'numpy'”. | No package install/download occurred. Smoke, mandatory REQUIRE_CUTILE=1 attention tests, CLI, benchmark and profile steps were all skipped. The 531-byte environment log was uploaded as [artifact 12146591326](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979187/artifacts/12146591326). |

### Related regression workflow records

- [CPU reference tests #39154979179](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154979179), commit b9189b5d: passed. These are hosted CPU/reference tests, not GPU/cuTile validation.
- [CPU reference tests #39154888102](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39154888102), commit 2971f5ac: passed. These are hosted CPU/reference tests, not GPU/cuTile validation.
- [A100 non-cuTile suite #39153449321](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39153449321), commit 4b5b0fbc: **908 passed, 52 skipped**, ordinary CUDA/reference coverage only.
- [CPU reference suite #39153449254](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39153449254), commit 4b5b0fbc: **764 passed, 68 skipped**, including GPU modules skipped on the hosted CPU runner.
- Earlier workflow attempts [#39152872387](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39152872387) and [#39152944500](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39152944500) created no jobs because the manual trigger was represented as a one-line array rather than the structural test's expected workflow_dispatch key. That contract was corrected later.

### Historical blocking dependency before the venv was created (superseded by 2026-10-19 execution)

The Windows runner is Online and sees the correct RTX 4060, but **the already-existing cuTile Python environment is not located under the repository root**. The global Python visible to the runner has no NumPy, CuPy, or cuTile. Under the no-download/no-install constraint, the workflow intentionally refuses to create a new environment or fetch Python packages.

**Required before more GPU work can run:** identify the already-installed Python executable that successfully ran scripts/cutile_smoke.py with CuPy and cuda.tile (or place that existing venv within this project directory). Once the existing environment is available to the runner without installing anything, rerun the native Windows workflow. There is currently no actual-cuTile attention correctness result, benchmark result, CLI result, or Nsight Compute profile from these runs. Do not mark completion until those experiments execute and their artifacts are preserved.


## 2026-10-19 — actual cuTile attention validated and benchmarked on RTX 4060

### Canonical run and artifact
- Successful workflow: [#39160827910](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910), commit 12120f8a67a1fbd6f9cd83bd8f2f8bf7b31ce428.
- Artifact: [12148414465](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39160827910/artifacts/12148414465), SHA-256 c84b781635a23571139ca8734df9e9d39ddab8485bfb5875c30db2ec63361854.
- The run logged identical WORKFLOW_SHA and CHECKOUT_GIT_HEAD. The project virtual environment contains CPython 3.12.10, NumPy 2.5.4, CuPy 14.2.0, cuda-tile 1.6.0 and matching TileIRAS/NVCC/NVVM 13.4.92. All pip/temp/CuPy/CUDA caches were project-local. No elevated privilege or system-wide toolkit change was made.

### Gates passed
1. Genuine cuTile smoke: vector-add kernel synchronized and exactly verified 4,096 outputs.
2. Actual attention suite: **48 passed in 7.05 s**, REQUIRE_CUTILE=1, zero failures.
3. Accuracy report: **29 legal case/tile configurations**, all matched the independent FP32 NumPy matrix oracle within the frozen mixed tolerance; zero violations. Max absolute error 0.0009765625; maximum tolerance ratio 0.0243665.
4. CLI: [1,1,64,32] FP16 seeded input; actual cuTile output passed shape/dtype/finiteness and independent-reference verification; CLI_REFERENCE_CHECK=PASS.
5. Performance: three sessions over seven frozen cases and five tile configurations. Ten warmups, 30 CUDA-event samples and five launches per sample. Baseline measured before/after each tile sweep; first-call latency recorded separately. All raw JSON/CSV and the aggregate summary are in the artifact.

### Current benchmark decision
- Long D=128, 64×64: median 0.267059 ms, P90 0.271626 ms, unfused CuPy 4.414310 ms, 16.53×; range 2.0%, maximum baseline drift 0.5%.
- Longer batched D=128, 64×128: median 0.923034 ms, P90 0.937656 ms, unfused CuPy 17.131469 ms, 18.54×; range 0.9%, maximum baseline drift 0.1%.
- Use 64×64 as the general default. The 64×128 specialization is promising for the longer batched D=128 case but should be independently rechecked before claiming a universal win. The 128×128 tile loses on long D=128. Short-case timing is highly noisy and not suitable for headline speedup claims.
- Optimizations retained in the fused kernel: skip wholly-future K/V tiles; use FP16 probabilities as P×V operands while retaining FP32 QK accumulation, softmax state, and output accumulator. The frozen correctness suite and numerical report passed afterward.

### Historical profiler blocker — superseded by the completed profiling gate below
- At the time of this earlier run, the workflow searched only project-local paths, so it reported `NCU_NOT_PRESENT_IN_PROJECT_DIRECTORY` and had no Nsight report artifact. This is historical and no longer describes the RTX 4060 state.
- After the user approved installing the official NVIDIA Nsight Compute Windows package, the updated native workflow found Nsight Compute 2026.3.1.0 and successfully profiled the RTX 4060. A separate follow-up workflow exported the reports and preserved them in artifact [12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039); see the completed profiling section below.
- The older A100 `ERR_NVGPUCTRPERM` event remains a distinct historical ordinary-CUDA issue. It did **not** occur during the RTX 4060 profiles.
- CUDA-enabled PyTorch SDPA comparison remains optional/unrun; the measured performance reference is the intentionally unfused FP32 CuPy implementation.


## 2026-10-19 — actual-cuTile Nsight Compute profiling completed

### Execution and artifacts
- Updated the native Windows cuTile workflow to find the installed Nsight Compute CLI under the standard NVIDIA installation location and profile only the intended steady-state launch. The repeated end-to-end native run [#39163083223](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163083223) succeeded: smoke, 48 required actual-cuTile tests, 29 numerical configurations, CLI/oracle, and all three controlled benchmark sessions passed.
- Nsight Compute CLI **2026.3.1.0 (build 38829034)** was found at the standard NVIDIA Program Files location. RTX 4060 hardware-counter collection succeeded; no `ERR_NVGPUCTRPERM` was observed.
- Captured basic throughput/resource profiles and supplemental scheduler/warp-state profiles for both `medium_d128` tile 64×64 and `longer_batched_d128` tile 64×128.
- Profile workflow: [#39163619066](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066). Complete report artifact: [12149902039](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39163619066/artifacts/12149902039). It contains four compressed `.ncu-repz` reports, their exported detail text/raw CSVs, and CLI logs/status.

### Measured profiler findings
| Metric | medium_d128 64×64 | longer_batched_d128 64×128 |
|---|---:|---:|
| Nsight kernel duration | 52.90 µs | 984.26 µs |
| Registers per thread | 231 | 255 |
| Static shared memory per CTA | 65.54 KiB | 81.92 KiB |
| Achieved occupancy | 8.35% | 8.33% |
| SM compute throughput | 18.81% | 30.27% |
| DRAM throughput | 22.88% | 11.89% |
| Scheduler cycles with no eligible warp | 85.83% | 85.26% |
| Warp cycles per issued instruction | 7.08 | 6.78 |
| Nsight rule execution-pipeline stall estimate | 2.4 cycles | 3.1 cycles |

The occupancy section identifies shared memory as the CTA-residency limiter (one CTA per SM). Nsight's estimated local speedup from the warp-state rule (33.57% and 46.04%) is a heuristic, not a measured speedup guarantee. The optional `smsp__pcsamp_sample_count` metric was unavailable, so source-level stall-sampling locations were not accessible.

### Decision and next-step boundary
- Mark the hard profiling gate complete. The dedicated [profile analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) discusses the evidence and limitations.
- Keep the validated 64×64 general default and the promising 64×128 longer-batched tile. The evidence points to shared-memory/register footprint and instruction scheduling as possible optimization areas, but no kernel change is justified without a new version passing correctness and repeated benchmarks.
- The optional CUDA-enabled PyTorch SDPA comparison remains outside the completed core scope. Current speedups are versus the deliberately unfused FP32 CuPy baseline, not a vendor-optimized attention kernel.
- Final evidence audit: the end-to-end native workflow #39163083223 and dedicated profiler workflow #39163619066 both completed successfully, and every required GPU/profile step reports success. Downloaded artifact 12149902039 contains four `.ncu-repz` files, their four detail exports and raw-metric CSVs; its downloaded ZIP SHA-256 matches GitHub's recorded digest `fd947285bc52731a97127fda08a86a9e53dc9a8000c9164966165fed0fc06b1c`.


## 2026-10-19 — assignment-level report and rubric closure audit

- Read and reconciled the original `nvidia2-cutile.pdf` requirements against current source, run logs, benchmark JSON/CSV and the four-profile Nsight Compute artifact.
- Prepared a seven-page, 5–8 page-compliant technical report for review outside the repository; the editable DOCX/PDF and rendered figures were inspected. Its Markdown source is [COURSE_PROJECT_REPORT_2026-10-19.md](COURSE_PROJECT_REPORT_2026-10-19.md).
- Added a requirement-by-requirement checklist at [COURSE_PROJECT_COMPLETION_CHECKLIST.md](COURSE_PROJECT_COMPLETION_CHECKLIST.md), including Week 1–4 milestones and the optional extension backlog from the assignment.
- The GPU technical gates remain PASS. Rubric closure remains PARTIAL: a CUDA-enabled framework/vendor attention comparison has not been measured; long-case clean-cache JIT fields are null because specializations were already compiled; a second profiler-guided optimization still requires an auditable before/after experiment; the individual contribution statement needs owner confirmation.
- No PyTorch or other extra package was installed. A framework reference needs explicit user approval before changing the project environment. The temporary workflow used only to export the source assignment PDF was deleted after the PDF was reviewed.
- Latest CPU/reference CI after temporary workflow cleanup: [run #39233514922](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39231291000), 764 passed, 68 skipped, 0 failed. The skips are expected in that CPU-only job and do not replace the native RTX 4060 evidence.

## 2026-10-19 — optional extension implementations and pending native validation

- Added optional GQA/MQA support by mapping query heads to grouped KV heads through a compile-time KV_GROUPS value; added an independent NumPy grouped-head reference and CUDA gate tests for Hkv=1 and Hkv=2.
- Added a packed-variable-length wrapper and independent reference with per-segment causal reset and a no-cross-sequence-leakage test. This wrapper deliberately launches once per segment; it is not described as a single-launch packed kernel.
- Added a correctness-gated tile autotuner which records all raw samples, requires three or more sessions, validates candidates before timing, and keeps the default tile when observed gain does not beat measurement spread.
- Added a PyTorch autograd bridge using the fused cuTile forward with an explicit FP32 reference backward, matched-weight QKV-projection model blocks, an SDPA reference block, GPU output/gradient tests, and an end-to-end model comparison benchmark. Its installation and GPU result remain pending; reference backward is not presented as a fused optimized backward.
- Added a DRAM load-latency hint sweep for 0/2/4/6/8/10 on the two profiler workloads, with NumPy correctness before timing, three order-rotated sessions, and a noise-aware keep/reject threshold. NCU basic captures for hint 4/8 are configured. No hint result has been measured yet.
- Added an actual Nsight Compute GUI window-capture helper using Windows PrintWindow on each native report; it rejects blank captures and records a manifest. The screenshot stage is only considered complete after real PNGs are captured and inspected. Existing SVGs are metric plots, not GUI screenshots.
- Because these changes touch the kernel build parameters and add GPU variants, the earlier 48/48 core GPU pass remains historical evidence for the former source baseline. The latest full native run must validate the new source before the core path and optional kernels are called revalidated.
- Remaining separate-dependency/alternative-track boundaries: BF16/FP8 need a dtype-support plan and likely ml_dtypes in the project venv (not installed without approval); CuTe DSL remains an instructor-approved alternative track.


## 2026-10-20 — current-source native validation, measured hint candidate, and PyTorch GEMM isolation

### Completed native RTX 4060 validation
- Completed [run #39422402438](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438) on commit `bf5a6233e195c2ced647b4e20b5b2d914da60e85`; raw artifact [#12264746195](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438/artifacts/12264746195), 194,711 bytes, SHA-256 `e3eaf4ae1e88d2547d2034dfe333db86b70f69e2f2d6c71b0d67b7c8bdde0c32`.
- Passed genuine cuTile smoke; 48/48 required core tests; 3/3 selected GQA/MQA and packed-varlen GPU tests; 29 legal numerical case/tile configs with no tolerance violation; independent CLI output oracle; three controlled benchmark sessions; all 36 accuracy-gated latency-hint trials; and two actual cuTile NCU captures.
- Exact summary in [CUTILE_RTX4060_RESULTS_2026-10-20.md](CUTILE_RTX4060_RESULTS_2026-10-20.md). Best long D=128 tile was 64×64 at 0.267059 ms median / 0.269128 ms P90 / 16.49×; best longer-batched D=128 tile was 64×128 at 0.915558 ms / 0.921457 ms / 18.69×. These ratios use the intentionally unfused FP32 CuPy baseline; no vendor-optimized attention comparison has passed.
- Benchmark controls: 10 warmups, 30 CUDA-event samples, 5 launches per sample, three sessions with rotated/reversed tile ordering. Accurate raw summary, all 29 tiles, first-call/JIT caveats and baseline bracket drift are recorded in the dated result file.

### Load-latency hint result and next optimization gate
- All 36 LOAD_LATENCY candidate/session runs passed the independent NumPy tolerance check before timing.
- Medium D=128 / 64×64: hint 10 median 0.053248 ms vs compiler default 0.058342 ms (8.73% gain); session ranges were 0.53% for both, so it passes the experiment's predeclared 1% noise-aware gate. Longer-batched D=128 / 64×128: best candidate hint 2 gained only 0.16% against a 5.88% threshold caused by observed spread; keep the default.
- Do **not** switch global default. The NCU workflow was amended to capture matching medium hint-10 metrics and raw exports; [run #39423368441](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423368441) remains queued. Inspect register/shared-memory/occupancy changes before exposing a workload-specific flag or declaring the Week 3 second optimization closed.

### PyTorch integration / cuBLAS diagnosis
- Expanded gradient coverage to batch/head layouts (1,1), (1,2), (2,1), (2,2). The CUDA 13.0 PyTorch wheel failed each `torch.bmm` inside the backward path with `CUBLAS_STATUS_INVALID_VALUE`.
- Added a plain float32 `torch.bmm` smoke before the cuTile bridge in [run #39423190840](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190840). It fails too, establishing that this is not solely a cuTile-backward/DLPack issue in the current environment.
- Added [scripts/torch_cublas_diagnostic.py](../scripts/torch_cublas_diagnostic.py), which runs 2D GEMM, batch-1 BMM, batch-2 BMM and FP16 linear in separate processes, and a workflow step to install the official PyTorch 2.14.0 CUDA 13.2 wheel into a *separate project-local venv*. This preserves `.venv-cutile-windows`; [run #39423804493](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423804493) is queued. Until it passes basic GEMM, no successful PyTorch gradients, model block, SDPA backend or framework timing are claimed.

### Runner interruption and documentation synchronization
- Repeat full native run [#39423190965](https://github.com/xemptynothing/gpu-nvidia/actions/runs/39423190965) passed environment setup, actual-cuTile smoke and core required tests, but stopped while the optional GQA/MQA/packed-varlen test step was shown as still in progress. The job has no retrievable log blob or artifacts; GitHub records a failed conclusion. Do not count it as a full extension failure or a full regression pass.
- At the latest API check, the Windows self-hosted jobs for NCU hint-10 and PyTorch CUDA 13.2 were queued and had not been picked up after the interrupted run. Ask the owner to confirm the Windows runner process/service remains online before claiming those results are about to complete.
- Updated [Project Two-Pager](PROJECT_TWO_PAGER.md), [Report.md](../Report.md), [README](../README.md), [execution evidence index](EXECUTION_EVIDENCE_INDEX.md), [course completion checklist](COURSE_PROJECT_COMPLETION_CHECKLIST.md), [TODO checklist](TODO_CHECKLIST.md), and [result status](../results/STATUS.md) to use the same current benchmark values and explicit blockers.
- Fresh docs-only CPU/reference workflows passed for commits `788befbe`, `9453d9f4`, `03542fe3`, and the current status/docs updates; those successes validate Python/Markdown-independent tests only and do not replace native GPU evidence.