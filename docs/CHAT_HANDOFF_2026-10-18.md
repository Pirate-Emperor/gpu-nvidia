# New-chat handoff — Fused Causal Attention with NVIDIA cuTile

**Prepared:** 2026-10-18  
**Repository:** [xemptynothing/gpu-nvidia](https://github.com/xemptynothing/gpu-nvidia) (private)  
**Branch:** main  
**Source snapshot reviewed before this documentation commit:** afe530bc090c779ebe5d741dcf7ae544985570d1  
**Purpose:** resume the technical project in a new chat without losing prior tests, benchmarks, failures, environment facts, or next steps.

> **Critical distinction:** the project's ordinary CUDA C++ kernels are well tested on A100. Actual NVIDIA cuTile execution has **not** been successfully validated. Never relabel surrogate results as cuTile results.

## 1. Project target and completion contract

Implement a fused **causal self-attention forward pass** with:

- Inputs Q/K/V stored as FP16, rank-4 [B,H,N,D], contiguous under the current contract.
- FP32 accumulation for Q·K, online-softmax recurrence, and weighted V accumulation; FP16 output.
- Scale by 1/sqrt(D), causal mask, numerically stable softmax.
- Required head dimensions D ∈ {32,64,128}.
- No global N×N score or probability matrix in the fused kernel; ideally one fused launch per output.
- Tile-divisible sequence sizes for the selected tile configuration; no partial edge tiles under current scope.
- Forward/causal only: no dropout, backward, BF16/FP8, GQA/MQA, KV cache, packed variable-length sequences, or unrelated DSL rewrites.
- At least two tile configurations, correctness against independent references, a deliberately unfused baseline and suitable framework/vendor baseline where available, at least 10 warmups and ≥30 CUDA-event samples, separate compile/JIT time, per-case median/P90/throughput, and successful profiles for at least two actual-cuTile configurations.

The actual target file is src/cutile_attention.py. Ordinary-CUDA implementations remain useful controls but cannot fulfil the actual-cuTile gate.

## 2. Current status

| Area | Current state | Evidence / boundary |
|---|---|---|
| Independent references | PASS | Full-matrix FP32 NumPy oracle, tiled online-softmax oracle, and independent scalar FP64 oracle. |
| Broad ordinary-CUDA/reference regressions | PASS | A100 non-cuTile run #52: 907 passed, 52 skipped, 0 failed. |
| Latest CPU/reference regression | PASS | Run #340 on pre-handoff main SHA afe530b: 763 passed, 68 skipped, 0 failed. |
| A100 CUDA fused variants | PASS | Native run #44: 133 passed, 1 skipped; separate v4 and v5 targeted tests each passed 30/30. |
| Actual NVIDIA cuTile compile/launch/correctness | **NOT VERIFIED / BLOCKED on old runner** | A100 runner driver is R550.90.07; current cuTile workflow requires R580+. |
| Actual cuTile performance comparison | OPEN | Cannot measure until actual-cuTile correctness passes. |
| Nsight Systems | PASS for ordinary-CUDA surrogate only | Two traces and CUDA API/kernel summaries uploaded; not a cuTile profile or a steady-state benchmark. |
| Nsight Compute hardware counters | **BLOCKED** | Previous attempt returned ERR_NVGPUCTRPERM; no successful kernel counter profile was captured. |
| User's personal/local GPU runner | NOT CONFIGURED YET | First collect GPU/driver/toolkit inventory, then choose a safe Windows-native or WSL2/Linux isolation strategy. |

### What counts as completion

Do not mark the project complete until a supported GPU really compiles and launches the actual cuTile kernel, the actual correctness matrix passes, the performance benchmark has controlled repeats and cold/JIT separation, and at least two actual-cuTile Nsight Compute profiles are inspectable. An import-only test, a skipped test, or a successful ordinary-CUDA test does not satisfy those gates.

## 3. Source map

| Path | Role |
|---|---|
| src/reference.py | Independent full-matrix FP32 NumPy oracle. |
| src/online_reference.py | Independent tiled online-softmax oracle; carefully handles wholly future causal tiles. |
| tests/test_reference_exhaustive.py | Independent scalar FP64 oracle and exhaustive mathematical checks. |
| src/baseline.py | Unfused CuPy score/probability-matrix baseline. |
| src/naive_cuda_attention.py | Deliberately slow scalar CUDA comparison path. |
| src/custom_cuda_attention.py | Block-row fused ordinary CUDA. |
| src/warp_cuda_attention.py | Warp-row fused ordinary CUDA. |
| src/shared_tile_cuda_attention.py | Current shared-memory tiled ordinary-CUDA surrogate, tile_k=16/32/64, four query rows per block, online softmax. |
| src/variants/shared_tile_v1.py … shared_tile_v5.py | Preserved optimization ladder; v4/v5 were tested but not promoted. |
| src/cutile_attention.py | Required actual-cuTile target; still needs a supported-driver end-to-end run. |
| tests/test_gpu_correctness.py, tests/test_gpu_edge_cases.py | Actual GPU/target correctness tests; must be run with cuTile truly available. |
| benchmarks/benchmark_v5.py | Three paired, alternating-order current-vs-v5 sessions for all seven cases × three tile sizes. |
| benchmarks/profile_shared_tile.py | Warmed single-launch profile target for ordinary CUDA. |
| .github/workflows/a100-nsight.yml | Actual-cuTile correctness gate followed by two intended Nsight Compute profiles on a compatible R580+ setup. |
| .github/workflows/a100-nsys-timeline.yml | Successfully captures ordinary-CUDA Nsight Systems traces and summary exports. |
| docs/TODO_CHECKLIST.md | Prioritized checklist and definition of done. |
| docs/TEST_MATRIX.md | Coverage ledger and run links. |
| docs/EXPERIMENT_REPORT.md | Approach history, failures, timing and resources. |
| results/STATUS.md | Runner inventory and status snapshot. |

## 4. Test coverage explained

The test strategy uses separate references to reduce correlated mistakes:

1. **Full-matrix FP32 NumPy oracle:** computes scores, scaling, causal mask, stable softmax, and weighted V in a direct, easier-to-inspect way.
2. **Tiled online-softmax reference:** validates the recurrence used by tiled kernels without allocating a global score matrix.
3. **Scalar FP64 oracle:** independent high-precision checks for the most sensitive small cases.

Input-contract tests check rank, shapes, dtype, contiguity, finite values, legal head dimensions and tile sizes. Mathematical/metamorphic tests include:

- Causal future-K/V invariance: changing keys/values after query position r must not change outputs up to r.
- Query locality: modifying query row r must not affect earlier rows.
- N=1 returns its only value vector.
- Zero logits yield the causal prefix mean of V.
- Constant V remains constant, independent of Q/K.
- Extreme finite logits must not create NaNs/Infs.
- Convex-hull/range invariant for each output dimension.
- Batch/head isolation, tiled-reference agreement, deterministic input generation and CLI I/O contract.

### Verified Actions runs

| Run | Result | What it validates | Link |
|---|---|---|---|
| A100 native CUDA #44 | 133 passed / 1 skipped | Ordinary CUDA RawKernel correctness and native CUDA path; not actual cuTile | [Run #44](https://github.com/xemptynothing/gpu-nvidia/actions/runs/37756646415) |
| A100 full non-cuTile #52 | 907 passed / 52 skipped / 0 failed | Broad non-cuTile regression (references, ordinary CUDA and project guardrails) | [Run #52](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38056264082) |
| CPU/reference #334 | 763 passed / 68 skipped | Earlier CPU/reference run; subsequently repeated | [Run #334](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38056264023) |
| CPU/reference #340 | 763 passed / 68 skipped / 0 failed | Latest CPU/reference suite; its head was the pre-handoff main SHA afe530bc090c779ebe5d741dcf7ae544985570d1 | [Run #340](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38056390931) |
| Isolated v4 A100 tests #1 | 30 passed | v4 differential/edge tests; not promoted | [Run #1](https://github.com/xemptynothing/gpu-nvidia/actions/runs/37760982099) |
| Isolated v5 A100 tests | 30 passed | v5 differential/edge tests; not promoted | [Run](https://github.com/xemptynothing/gpu-nvidia/actions/runs/37761585734) |
| Current-vs-v5 paired benchmark #4 | Correctness 30/30 + 3 paired sessions for each case/tile | Order alternates by paired session; see numerical table below | [Run #4](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055001163) |
| PTXAS report #5 | v1/v2/v3/current/v4/v5 compiled for SM80; zero reported spills | Compile/resource information, not proof of runtime occupancy or speed | [Run #5](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38054963187) |
| Nsight Systems timeline #2 | Two .nsys-rep traces + readable CUDA API/kernel summaries | Ordinary-CUDA surrogate timeline only | [Run #2](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055834950); [artifact](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954) |
| Nsight Compute #1 | FAILED before useful profiling | ERR_NVGPUCTRPERM; no actual profile collected | [Run #1](https://github.com/xemptynothing/gpu-nvidia/actions/runs/37760475268) |

**Run/commit caveat:** the broad A100 #52 run is the latest verified broad GPU regression cited here, but it ran before the last docs-focused main commits; the CPU/reference #340 run ran on the pre-handoff main SHA noted above. This documentation handoff itself does not modify kernel source. Re-run the relevant test suite on the new main head when CI is available rather than assuming documentation changes were GPU-tested.

## 5. Current-vs-v5 paired benchmark results

**Method:** run #4 measures seven representative cases × tile_k ∈ {16,32,64} (21 combinations). Each paired session times both implementations; each kernel run has 10 warmups and 30 CUDA-event samples. Three paired sessions per case/tile alternate order. Latencies below are in milliseconds; ratio is the median paired current latency / v5 latency. Thus, ratio > 1 means v5 was faster in the paired measurement. Since it is a paired statistic, it need not equal the ratio of the two displayed marginal median latencies. Results are from **ordinary CUDA**, not cuTile.

| Case | tile_k | Current median (ms) | v5 median (ms) | Median paired current/v5 | Range across paired ratios |
|---|---:|---:|---:|---:|---:|
| short_d32 | 16 | 0.050 | 0.051 | 0.980× | 0.960–0.980× |
| short_d32 | 32 | 0.064 | 0.052 | 1.276× | 0.971–1.402× |
| short_d32 | 64 | 0.049 | 0.050 | 0.980× | 0.980–1.353× |
| short_d64 | 16 | 0.075 | 0.058 | 1.304× | 1.000–1.465× |
| short_d64 | 32 | 0.057 | 0.057 | 0.982× | 0.794–1.018× |
| short_d64 | 64 | 0.057 | 0.061 | 0.933× | 0.775–1.018× |
| short_multihead_d64 | 16 | 0.085 | 0.085 | 0.976× | 0.950–1.012× |
| short_multihead_d64 | 32 | 0.084 | 0.093 | 0.906× | 0.764–1.232× |
| short_multihead_d64 | 64 | 0.082 | 0.084 | 0.988× | 0.947–1.000× |
| medium_d64 | 16 | 0.279 | 0.251 | 1.110× | 1.020–1.110× |
| medium_d64 | 32 | 0.276 | 0.254 | 1.116× | 0.909–1.123× |
| medium_d64 | 64 | 0.280 | 0.247 | 1.023× | 1.015–1.135× |
| medium_d128 | 16 | 0.933 | 0.908 | 1.024× | 1.018–1.028× |
| medium_d128 | 32 | 0.872 | 0.829 | 0.993× | 0.989–1.083× |
| medium_d128 | 64 | 0.807 | 0.806 | 1.002× | 0.993–1.002× |
| long_d128 | 16 | 5.257 | 5.239 | 1.003× | 1.003–1.004× |
| long_d128 | 32 | 5.192 | 5.170 | 1.004× | 1.004–1.005× |
| long_d128 | 64 | 5.152 | 5.129 | 1.004× | 1.003–1.005× |
| longer_batched_d128 | 16 | 20.421 | 20.368 | 1.004× | 0.996–1.004× |
| longer_batched_d128 | 32 | 20.172 | 20.065 | 1.004× | 1.004–1.008× |
| longer_batched_d128 | 64 | 20.017 | 19.902 | 1.006× | 1.003–1.007× |

### Decision from this benchmark

- v5 (half2 QK calculation) passes its targeted correctness tests.
- It has a modest improvement on some medium-D=64 combinations, mixed results for short cases, and only around 0.3–0.6% on long/longer-batched cases.
- The paired ranges can be noisy. **Do not promote v5** on this evidence.
- v4 also remains unpromoted: no strong repeatable win and PTXAS/cubin summary was identical to current for the reported resource summary.
- Do not infer a single broad speedup from the fastest short-case observation.

## 6. PTXAS compile-time resource report

From PTXAS workflow #5 (SM80):

| Variant | Registers/thread | Static shared memory | Spill stores/loads | Cubin bytes |
|---|---:|---:|---:|---:|
| v1 | 28 | 33,544 B | 0 / 0 | 11,040 |
| v2 | 32 | 33,792 B | 0 / 0 | 15,392 |
| v3 | 40 | 33,792 B | 0 / 0 | 22,560 |
| current | 32 | 33,792 B | 0 / 0 | 15,136 |
| v4 | 32 | 33,792 B | 0 / 0 | 15,136 |
| v5 | 32 | 33,792 B | 0 / 0 | 15,520 |

No reported spills is useful compile-time evidence, but it is **not** a substitute for runtime performance, occupancy, or hardware-counter profiling.

## 7. Nsight Systems summary and interpretation

Successful surrogate timeline: [run #2](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055834950), artifact [11671075954](https://github.com/xemptynothing/gpu-nvidia/actions/runs/38055834950/artifacts/11671075954).

- Case 1: medium_d128, tile_k=64. Seven kernel instances per trace (initial call/JIT, five warmups, final profiled call); instrumented trace duration median **0.977 ms**, mean **0.998 ms**, range **0.975–1.126 ms**.
- Case 2: longer_batched_d128, tile_k=32. Seven kernel instances per trace; instrumented trace duration median **25.851 ms**, mean **25.925 ms**, range **25.845–26.369 ms**.
- CUDA API summary included significant allocation activity, including cudaMalloc.
- These are **traced-run durations**, not steady-state benchmark latency. They include first-call/JIT and profiler/tracing overhead, so do not compare them directly with the CUDA-event latency table.
- They are ordinary shared-tile CUDA traces, not actual cuTile profiles.

Nsight Compute remains blocked by ERR_NVGPUCTRPERM. Do not attempt privileged profiling workarounds. Use an explicitly authorized profiling environment or ask the responsible admin to enable the approved counter access.

## 8. A100 runner / toolchain facts already verified

Runner used for the successful A100 experiments:

- Runner: gpu-project; machine: scn21-mn; runner version 2.337.0.
- OS: Ubuntu 22.04 family; kernel 5.15.0-1062-nvidia; glibc 2.35; x86_64.
- Physical host: 8 × NVIDIA A100-SXM4-40GB plus 6 second-generation NVSwitches.
- Visible to this runner: GPU 0 and 1, both full A100-SXM4-40GB; CUDA_VISIBLE_DEVICES=0,1; SM80 / compute capability 8.0; 40,960 MiB each; MIG disabled; ECC enabled.
- Driver R550.90.07; system CUDA Toolkit/NVCC 12.4.131; GCC/G++ 11.4.0.
- Nsight Compute 2024.1.1.0; Nsight Systems 2023.4.4.54; Fabric Manager installed and active.
- Runner process is non-root. Sudo is not passwordless. The workflow cannot upgrade the host driver.
- Core config is at /nlsasfs/home/lmedge/rahkus/OEC/core_config.py; install steps that need the configured proxy use python3 scripts/with_core_config.py <command> ... and do not store proxy credentials in the repository.

### Tried and not validated

- cuda-compat-13-4 extracted in user space, but CUDA failed with cudaErrorInsufficientDriver; loading compatibility libraries then produced cudaErrorSystemDriverMismatch (error 803).
- CUDA 13.0 forward-compatibility attempt did not provide an end-to-end actual-cuTile run.
- An attempt to install cuda-tile 1.2.0 could not obtain that package from the configured Python index.
- Actual cuTile workflow intentionally gates on R580+; detailed cuTile Nsight Compute stats require R580.126.09+ per existing project environment notes.
- Do not repeat these attempts without a materially different, supported environment.

## 9. Local GPU inventory and secure project-only runner plan

The user has **not yet supplied local GPU specs** and the local runner has **not** been set up. First collect the actual machine/toolchain output.

### Quick commands

Run in Windows PowerShell:

    nvidia-smi
    nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total,memory.free,pci.bus_id --format=csv
    nvcc --version

If a query field is not supported on that driver, use nvidia-smi -q. In a Linux/WSL terminal, the same commands apply if NVIDIA utilities and the CUDA Toolkit are installed.

The top-level CUDA version printed by nvidia-smi is the driver-supported CUDA level; it is not proof that that CUDA Toolkit is installed. nvcc --version reports the installed compiler/toolkit if present.

### Security objective

The runner should execute the **specific project's reviewed workflow**, not provide ChatGPT with an unrestricted interactive terminal on the user's computer. A runner executes repository workflow code on the host/container; therefore any workflow code has the OS permissions available to the runner account.

Recommended policy:

1. Use a disposable Linux VM or an isolated GPU-enabled container on a dedicated machine where practical. Choose the exact Windows-native vs WSL2/Linux method only after seeing GPU/driver/OS output.
2. Create a dedicated non-admin/non-root account and a one-project workspace. Only that workspace and required temporary/build paths should be writable or mounted.
3. Do not mount the personal home directory, .ssh, browser profiles, credential stores, cloud configuration, unrelated drives, or other private project directories.
4. Never run a privileged container; do not expose the host Docker socket, host PID namespace, or host root. For containers, drop unnecessary capabilities, enable no-new-privileges, use a restrictive seccomp/AppArmor profile, make the container root filesystem read-only where practical, and use narrowly scoped writable temporary paths. An isolated VM is a stronger boundary when feasible.
5. Enable only the NVIDIA GPU devices/driver interface and compute libraries needed for tests. Do not grant sudo, Windows administrator membership, CAP_SYS_ADMIN, host-management, reset, or driver-management privileges.
6. Keep personal PATs, SSH keys, passwords, cloud credentials, and unrelated secrets out of the runner and job environment. Use workflow permissions no broader than needed (start with contents: read). Avoid secrets entirely in test jobs if possible.
7. Register the runner only to this private repository. Begin with manual workflow_dispatch jobs on trusted main; never execute untrusted pull-request code on the local GPU runner. Prefer ephemeral/disposable one-job runners and destroy/recreate the execution environment after the job.
8. Allow only required outbound HTTPS (GitHub and approved package/dependency endpoints), disable inbound remote shell, and log runner lifecycle, workflow SHA, and test results.
9. Test the isolation before compiling the project: verify GPU visibility and CUDA work, sudo is unavailable, only the intended workspace is writable, personal paths/credentials are not mounted, and the job cannot access the host container daemon.

**Authentication nuance:** GitHub Actions runner registration/connection authentication is mandatory, so zero GitHub authentication is not possible. Keep the runner registration credentials on the machine, do not provide a personal PAT or admin credentials to workflow steps, never paste registration tokens into chat, and never commit runner config/credential files. GitHub's workflow token should have minimal explicit permissions. Ephemeral execution reduces persistence, but does not make arbitrary workflow code safe by itself.

ChatGPT does not receive a remote shell or direct filesystem access simply because the runner is connected; GitHub Actions jobs run on the machine. We can help author reviewed workflow code and inspect the logs/results provided back by GitHub.

## 10. Prioritized next objectives

### Step 1 — inventory the local machine (do first)

- [ ] Collect nvidia-smi, the detailed GPU query, and nvcc --version output.
- [ ] Confirm OS (Windows native, WSL2, or Linux), Python version and whether Docker/WSL2 is installed.
- [ ] Pick the narrowest practical isolation model. Do not register the runner until its mounts, account, privileges, secrets policy and workflow trigger policy are agreed.

### Step 2 — create and validate the isolated runner

- [ ] Use one-project writable mount and no personal directories/credentials.
- [ ] Make the runner account non-admin with no sudo; avoid host Docker socket and privileged mode.
- [ ] Register to this private repository only and use trusted/manual workflows initially.
- [ ] Add a read-only environment inventory job and a boundary smoke test. Never print tokens or credential files.

### Step 3 — establish actual cuTile support

- [ ] Record GPU model, compute capability, driver, toolkit, Python, package versions and environment variables.
- [ ] Import actual cuTile, compile a tiny kernel, launch it, synchronize and compare a known output. Preserve stdout/stderr and tested commit.
- [ ] If the local GPU/driver does not meet the installed cuTile requirements, report the gate accurately; do not use a CUDA-like ordinary kernel as a substitute.

### Step 4 — validate the attention target

- [ ] Run tests/test_gpu_correctness.py and tests/test_gpu_edge_cases.py with actual cuTile available.
- [ ] Validate D=32/64/128, at least two legal tile configurations, mixed absolute/relative error, causal future-K/V invariance, stability, input contract and edge behavior.
- [ ] Run python run_attention.py --input INPUT.npz --output OUTPUT.npy --backend cutile; validate shape, FP16 dtype, finite output and agreement with the independent oracle.
- [ ] Record the workflow URL, exact SHA, GPU/toolchain, pass/skip/fail counts and artifacts. No successful test should be inferred from skips.

### Step 5 — benchmark and profile actual cuTile

- [ ] Compare against unfused CuPy and best validated ordinary-CUDA control; add PyTorch SDPA only if installed and validated.
- [ ] Use ≥10 warmups and ≥30 CUDA-event measurements; report median/P90/throughput, and measure cold/JIT separately.
- [ ] Repeat with alternating/interleaved order and comparable GPU load; preserve raw JSON/CSV.
- [ ] Obtain authorized Nsight Compute counter access and save at least two real actual-cuTile profiles, one medium and one long/batched.
- [ ] Optimize only after correct baseline, reproducible timing and profiles exist. Keep v4/v5 as unpromoted ordinary-CUDA experiments unless new evidence changes the decision.

### Step 6 — close the reporting loop

- [ ] Update README.md, docs/TODO_CHECKLIST.md, docs/TEST_MATRIX.md, docs/EXPERIMENT_REPORT.md, docs/PROGRESS.md, and results/STATUS.md after material experiments.
- [ ] Keep actual-cuTile and ordinary-CUDA results in separate clearly labelled tables.
- [ ] Do not mark completion until actual-cuTile correctness, controlled performance comparison, and two successful profiles have evidence.

## 11. Working rules for the next chat

- Inspect the current main HEAD and the handoff docs before editing; do not assume main still equals the source SHA listed above because this handoff commit itself changes documentation.
- Use the connected GitHub repository tools for repo operations; commit changes to main for this project unless there is a reason to isolate risky work. Do not overwrite old experimental variants.
- Run actual tests and benchmarks rather than only making a plan. Give concise progress updates during longer work.
- Do not fabricate local GPU inventory, cuTile success, Nsight Compute metrics, speedups, test counts or artifact links.
- Prefer small reproducible experiments, preserve logs/artifacts, and report failures and regressions alongside wins.
- In the final report, include TEST / WHY / EXPECTED / ACTUAL / DECISION/NEXT.
- Never ask for sudo passwords, GitHub PATs, runner registration tokens, proxy passwords, or other credentials. Explain if a local user action is required.
- **First action in a new chat:** read this file and docs/TODO_CHECKLIST.md, verify current main, then proceed with the local GPU inventory/security boundary rather than reopening already-settled history.

## 12. Continuation audit — 2026-10-18

- At session start, the repository's `main` HEAD was `e87d8b9352fdf6783347d5c1e1945172011ebeb9`, matching the handoff commit.
- The actual cuTile kernel, independent references, CUDA variants, GPU tests, benchmark scripts and workflows were inspected. No actual-cuTile compile/launch/correctness result was produced in this continuation; no local GPU inventory was supplied.
- Route decision: do not use the existing R550.90.07 A100 as the actual-cuTile path. Keep it for ordinary-CUDA surrogate testing. Use the local GPU only after confirming its GPU model/compute capability, driver, OS, Python, Docker and NVCC/toolkit, or another explicitly authorized environment meeting NVIDIA's current cuTile support requirements. Do not assume a specific GPU or register a runner before testing the OS-enforced boundary.
- Official Quickstart check: https://docs.nvidia.com/cuda/cutile-python/quickstart.html states Linux x86_64, Linux aarch64 and Windows x86_64 support; compute capability 8.x–12.x; driver R580+; and Python 3.10–3.14. The optional [tileiras] packages require matching major.minor versions. Detailed profiling requires R580.126.09+ on Linux or R582.16+ on Windows.
- Actions re-audit limitation: The connected GitHub Actions run-list endpoint returned zero visible runs; direct run-detail/job/artifact requests for the prior handoff run IDs returned 404. The current handoff commit also had no visible combined status. This is an access/visibility limitation, not proof historical runs do not exist; previous pass/skip counts are retained as handoff records and were not independently re-audited in this continuation.
- The continuation updated routing and evidence notes in README, TODO checklist, test matrix, experiment report, environment/admin notes, progress log and current result status. No test execution or profiler capture was claimed. Actual-cuTile correctness, benchmark and profile gates remain open.

## 13. New local GPU inventory and native Windows build-up

The user supplied the actual local machine output after the prior handoff was written:

- NVIDIA GeForce RTX 4060, compute capability 8.9, driver 616.92, CUDA UMD 13.4.
- 8,188 MiB total VRAM; 7,925 MiB free at inventory capture.
- Windows 10 Pro x64, build 26300, WDDM.
- `nvcc` unavailable; Python is not installed/available through the `python` command; WSL2 and Docker are not installed.

NVIDIA's [current cuTile Python Quickstart](https://docs.nvidia.com/cuda/cutile-python/quickstart.html) explicitly lists Windows x86_64, compute capability 8.x–12.x, R580+ and Python 3.10–3.14. The reported GPU, driver and OS therefore pass those hardware gates on paper. Native Windows is the chosen first path; install per-user 64-bit CPython 3.12 and use `docs/LOCAL_GPU_WINDOWS.md`.

The repository now has `scripts/cutile_smoke.py` (a real cuTile vector-add compile/launch/synchronize/output check) and `scripts/setup_cutile_windows.ps1` (project-local venv using `cuda-tile[tileiras]` and `cupy-cuda13x[ctk]`). The two primary GPU test modules recognize `REQUIRE_CUTILE=1` so a missing dependency fails instead of turning the acceptance run into a skip.

**Status:** these scripts have not run on the user's Windows machine. Actual cuTile compile/launch, attention correctness, benchmark and profiles remain unverified. Do not register a GitHub runner on the personal Windows host before the manual smoke/correctness gates and a separate OS-enforced least-privilege boundary are validated. A Windows NVRTC builtins DLL issue was reported for an older stack (see [NVIDIA issue #57](https://github.com/NVIDIA/cutile-python/issues/57)); this is another reason the genuine smoke test is a hard gate rather than inferring success from the support matrix.
