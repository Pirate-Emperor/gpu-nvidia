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

The immediate next task is to provide the actual local GPU inventory. Do not assume the GPU model, driver, OS or toolkit. Run these commands in Windows PowerShell, or the equivalent terminal on Linux/WSL2:

    nvidia-smi
    nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,memory.free,pci.bus_id --format=csv
    nvcc --version

First use the currently documented facts; do not invent my local GPU specs or assume the OS/driver/toolkit. Help choose Windows-native vs WSL2/Linux after the output is available. Design a secure, project-only GitHub Actions runner:
- dedicated non-admin/non-root account;
- preferably disposable VM or tightly isolated GPU-enabled container;
- only this project's workspace writable/mounted;
- no sudo/administrator membership, no privileged container, no host Docker socket, no personal home/SSH/browser/cloud credential mounts, no unrelated files;
- minimal outbound network, no inbound remote shell, only the needed GPU compute interface;
- private-repository runner, trusted/manual workflows first, no untrusted pull-request execution, minimal workflow token permissions, ephemeral/disposable lifecycle.

The GitHub runner must authenticate to receive jobs, but registration/connection credentials must stay local and must never be pasted into chat or exposed to workflow code. Do not request passwords, PATs, runner tokens, or sudo credentials. Be clear that connecting an Actions runner does not give ChatGPT a live shell; Actions workflows execute on that runner.

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

**Start now** by reading the handoff and inspecting the current repository head/workflows; then determine whether the safe local GPU runner can be configured from the actual machine inventory. If the machine inventory has not yet been supplied in this conversation, continue work that does not depend on guessing it and state exactly which local-only input is still needed.
