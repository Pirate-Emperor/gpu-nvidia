# Prompt for the next ChatGPT conversation

Continue the ongoing project in the private GitHub repository **xemptynothing/gpu-nvidia**.

**Before doing anything else, read these repository documents from main:**
1. docs/CHAT_HANDOFF_2026-10-18.md
2. README.md
3. docs/TODO_CHECKLIST.md
4. docs/EXPERIMENT_REPORT.md
5. docs/TEST_MATRIX.md
6. results/STATUS.md

Do not ask me to repeat project history or known results. First inspect the current main HEAD because the handoff documentation commit will advance it.

## Primary objective

Complete the actual NVIDIA cuTile fused causal attention implementation—not just ordinary CUDA surrogates. Follow the project's frozen scope: FP16 Q/K/V, FP32 accumulation, causal online softmax, D=32/64/128, no global N×N score/probability matrix, at least two tile configurations, rigorous independent-reference correctness, measured comparisons, separate JIT timing, and two successful actual-cuTile hardware profiles.

## Immediate local-GPU/runner task

The user's actual local inventory has been supplied:

- NVIDIA GeForce RTX 4060, compute capability 8.9, driver 616.92, reported CUDA UMD 13.4, 8,188 MiB total / 7,925 MiB free VRAM at capture.
- Windows 10 Pro x64, build 26300, WDDM.
- System `nvcc` remains unavailable; CPython 3.12.10 and cuTile 1.6.0 are installed in the project venv. WSL2 and Docker Desktop are not installed yet.

NVIDIA's current cuTile Python Quickstart lists Windows x86_64, CC 8.x–12.x, driver R580+ and Python 3.10–3.14. The genuine cuTile vector-add smoke now passes locally (`GPU_SYNCHRONIZE=PASS`, `OUTPUT_CHECK=PASS`, `CUTILE_SMOKE=PASS`). The actual attention kernel remains unvalidated. The existing `gpu-project` A100/R550 runner is only an ordinary-CUDA control; do not use it as the actual-cuTile route.

New repository tooling:
- `scripts/setup_cutile_windows.ps1` creates `.venv-cutile-windows` and installs `cuda-tile[tileiras]` + `cupy-cuda13x[ctk]` without driver/system-Toolkit changes.
- `scripts/cutile_smoke.py` compiles, launches, synchronizes and verifies a genuine cuTile vector-add kernel; this has passed locally.
- Docker runner setup is now implemented: `docker/cutile-runner/`, `.github/workflows/cutile-rtx4060-docker.yml`, and `scripts/start_cutile_docker_runner.ps1`. The launcher checks GPU passthrough before requesting a short-lived token, registers an ephemeral one-job container, dispatches the workflow on main, watches its result, and cleans up.
- Next local prerequisite: install/enable WSL2, install Docker Desktop and GitHub CLI, authenticate `gh`, then run `scripts/start_cutile_docker_runner.ps1` once. The workflow itself runs the strict actual attention correctness/edge tests with `REQUIRE_CUTILE=1`.
- Container isolation reduces host exposure but is not a perfect security boundary. The runner has no host mounts or Docker socket, runs non-root with dropped capabilities, and only the trusted main-branch workflow is eligible. Runner registration credentials must stay local and never be exposed to job steps or pasted into chat.

## Known technical state — do not lose this

- Available project runner: gpu-project on scn21-mn; two full visible A100-SXM4-40GB GPUs (SM80/CC 8.0), R550.90.07, CUDA Toolkit 12.4.131, GCC 11.4.0, Nsight Compute 2024.1.1.0, Nsight Systems 2023.4.4.54, Fabric Manager active, non-root, no passwordless sudo.
- Actual cuTile target src/cutile_attention.py has not completed a verified compile + launch + correctness run. NVIDIA's current Quickstart requires R580+; detailed profiling requires R580.126.09+ on Linux or R582.16+ on Windows. Stop using the R550.90.07 A100 as the actual-cuTile route, but keep it for ordinary-CUDA surrogate tests. Use the inventoried local GPU or another explicitly authorized supported environment.
- CUDA 13.4 compatibility attempt failed with insufficient-driver and then system-driver-mismatch/error 803; CUDA 13.0 attempt produced no verified actual-cuTile run; the cuTile 1.2.0 package wasn't reachable in the configured package index. Do not repeat these without a materially different supported environment.
- Latest broad A100 ordinary-CUDA/non-cuTile regression: **907 passed / 52 skipped / 0 failed**, [run #52](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38056264082).
- Latest CPU/reference suite on pre-handoff main SHA afe530bc090c779ebe5d741dcf7ae544985570d1: **763 passed / 68 skipped / 0 failed**, [run #340](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38056390931).
- Native CUDA earlier: **133 passed / 1 skipped**, [run #44](https://github.com/xemptynothing/gpu-nvidia/actions/runs/37756646415).
- Shared-tile v4 and v5 each passed 30 targeted tests but remain unpromoted. The paired current-vs-v5 benchmark shows mixed short-case results and only ~0.3–0.6% long/batched gains. Keep the 21-combination result table from the handoff and do not claim a general speedup.
- PTXAS compiled v1/v2/v3/current/v4/v5 for SM80 with zero reported spills; v5 cubin 15,520 B vs current 15,136 B. This is compile-time evidence only.
- Nsight Systems successfully captured two **ordinary-CUDA surrogate** traces and summaries: [run #2](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055834950), [artifact](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954). Trace durations include JIT/profiling overhead and are not steady-state benchmark latencies.
- Nsight Compute hardware counter collection is blocked by ERR_NVGPUCTRPERM; no useful profile was captured. Do not use privileged workarounds.
- During the 2026-10-18 continuation, repository contents were readable but Actions run-list returned zero visible runs and historical run-detail/jobs/artifacts returned 404. Prior CI counts remain historical handoff records, not newly reverified evidence.
- Relevant files and detailed historical benchmark values are in docs/CHAT_HANDOFF_2026-10-18.md.

## Work style and acceptance requirements

Do real experiments; keep progress updates concise; run targeted correctness before broad regressions; preserve v1–v5 and all failed/negative results; do not overclaim. Use the GitHub connector to inspect/edit this repository; push documentation/code commits to main for this project unless risky work requires an isolated branch. Do not overwrite historical variants. Update README/TODO/test matrix/progress/status as evidence changes. Never assume a test is successful because it was skipped.

End technical updates with:

    TEST:
    WHY:
    EXPECTED:
    ACTUAL:
    DECISION/NEXT:

**Start now** by inspecting the current main HEAD and the new Docker runner files. The local smoke gate has passed. The immediate next gate is the one-time host prerequisite installation (WSL2 + Docker Desktop + GitHub CLI), then run `scripts/start_cutile_docker_runner.ps1` to automatically launch the ephemeral runner and dispatch/watch the real attention correctness workflow. Do not claim attention correctness, benchmark success, or profiler completion until those runs actually produce evidence.
