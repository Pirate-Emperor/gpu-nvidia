# Mentor demo execution guide

Target: the native Windows RTX 4060 machine and the project-local cuTile environment. Run these commands from the repository root unless a command explicitly uses a full path.

This guide describes the scripts as they should be run on the current repository. Test counts are collected from pytest at runtime; labels such as “48 tests” or “34 + 14” are historical and must not be treated as a current contract.

## 1. Start the interactive menu

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1
~~~

The menu resolves the repository root from its own script location and changes into that directory for the duration of the menu. It does not depend on the PowerShell process having started in the repository root.

| Menu item | Purpose |
|---|---|
| 1 | Standard ten-step walkthrough: GPU preflight, runtime smoke, correctness, CLI, numerical matrix and optional experiments |
| 2 | Guided kernel tour: displays source ranges, explains the invariant under test, then runs selected examples and every currently collected GPU test one by one |
| 3 | Collects and runs the complete required native cuTile GPU suite |
| 4 | Lists the currently collected test node IDs and runs one selected test |
| 5 | Executes every legal workload/tile pair in the JSON registry against the independent FP32 NumPy reference |
| 6 | Runs the three-session tile timing study |
| 7 | Runs the three-session load-latency-hint study |
| 8 | Runs CPU-only unit tests for the editable configuration registry |
| 9 | Opens the interactive registry editor to add a workload or tile candidate |
| 10 | Validates and lists the registry without launching the GPU |
| 11 | Opens docs/SUBMISSION_REPORT_2026-11-03.md |
| 12 | Runs the reusable mentor demo end to end |
| 13 | Runs the mentor demo with a live Nsight Compute capture attempt |

The guided and performance options are deliberately separated: the guided tour has pauses and explains code/test intent; the reusable demo is shorter and is suitable for a repeatable end-to-end execution.

## 2. Direct commands

Standard walkthrough, with prompts:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Standard
~~~

Guided source tour, with prompts:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Guided
~~~

Guided tour without pauses, but skipping the longer timing sweeps:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Guided -NoPause -SkipTileExperiment -SkipLatencyHintExperiment
~~~

Reusable end-to-end demo:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_demo.ps1
~~~

Quick demo plus Nsight Compute attempt, leaving the longer tile sweep out:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_demo.ps1 -SkipTileSweep -WithProfile
~~~

NoPause is intended for automated validation or an uninterrupted run. For a live presentation, omit it so the source explanation can be read before each test starts.

## 3. Standard walkthrough: what each step does

### Step 1 — Environment and GPU preflight

The script prints the repository root and the exact Python executable, then records the Python version and nvidia-smi inventory. The expected device for this project run is the NVIDIA GeForce RTX 4060 (compute capability 8.9). This checks that the experiment is not accidentally running under a system Python, a different virtual environment, or a CPU-only session.

**Expected evidence:** Python version, GPU model, driver version, compute capability and memory inventory. Stop if the intended GPU or project venv is not found.

### Step 2 — Genuine cuTile runtime smoke

scripts/cutile_smoke.py compiles and launches a minimal cuTile kernel and synchronizes the device. This isolates compiler, package, CUDA-runtime and device-launch problems from attention-kernel correctness. It is a runtime check, not an attention benchmark and not proof of numerical correctness for the full attention implementation.

**Expected evidence:** OUTPUT_CHECK=PASS and CUTILE_RUNTIME_CHECK=PASS.

### Step 3 — Core correctness suite

Pytest first collects the concrete test node IDs from tests/test_gpu_correctness.py. The script prints that count and then runs the file using REQUIRE_CUTILE=1, which prevents an unavailable cuTile dependency from silently turning a required GPU test into a skip.

The core file includes the workload/tile comparisons against the independent reference, additional supported tile coverage, small/large-logit numerical stability, causal future-value invariance and rejection of a non-divisible sequence.

**Expected evidence:** pytest summary with zero failures, plus REQUIRED_GPU_TESTS=... PASSED after the edge suite is also complete. The number is dynamic because parametrized test cases and the registry can change.

### Step 4 — Edge-case suite

Pytest separately collects and runs tests/test_gpu_edge_cases.py. These tests are not merely duplicates of random numerical comparisons. They cover properties that should hold for specially constructed data:

- Constant value vectors must remain constant under a normalized weighted average.
- Zero query/key logits produce uniform probability over each allowed causal prefix; the result must equal that prefix average.
- Different legal tile shapes must agree with the independent reference.
- Mutating future keys or values must not change outputs for earlier query positions.
- NaN and infinity, unsupported dimensions, non-contiguous inputs and sequence lengths that violate tile divisibility must be rejected.
- A value dimension Dv different from query/key dimension D must still produce the expected output shape and numerical result where supported.

**Expected evidence:** pytest summary with zero failures. The script reports the counts of collected core and edge test nodes rather than relying on old fixed numbers.

### Step 5 — Generate deterministic input

scripts/generate_inputs.py creates a reproducible NPZ containing Q, K and V with batch 1, one attention head, sequence length 64 and head dimension 32. The seed is 20261018. The generated file is retained under .tmp/logs/ using the run timestamp.

Reproducibility matters: the CLI and independent verifier consume this exact saved input, rather than generating separate random tensors.

**Expected evidence:** the input NPZ exists at the printed path.

### Step 6 — Run the course-facing CLI

run_attention.py reads the saved NPZ and invokes the real cuTile backend with query/key tile sizes 64×64. This exercises the public command-line path, file loading, backend selection, kernel launch, synchronization and result writing—not just a direct Python function call.

**Expected evidence:** the CLI returns exit code zero and creates the requested NPY output.

### Step 7 — Independent NumPy verification

scripts/verify_cli_output.py reloads the input and output files, computes the independent CPU reference and checks the saved result. The verifier checks output shape, FP16 dtype, finite values and the project's mixed absolute/relative numerical tolerance. The kernel output is not used to construct its own expected answer.

**Expected evidence:** CLI_REFERENCE_CHECK=PASS (or the equivalent pass marker emitted by the verifier). A present output file alone is not sufficient.

### Step 8 — Legal numerical configuration matrix

scripts/legal_config_registry.py validate checks the JSON registry before execution. scripts/mentor_numerical_matrix.py then enumerates every case/tile pair whose query and key/value tile sizes both divide the sequence length. Every legal pair is launched on the GPU and compared with the independent full-matrix FP32 NumPy reference.

The comparison uses the mixed envelope:

~~~text
abs(actual - reference) <= 0.02 + 0.02 * abs(reference)
~~~

The script writes a JSON summary and a sibling CSV. It reports expected legal pairs, checked pairs, pass count and fail count. A completed command is not enough: the JSON must report every legal pair passed and zero failures.

The current registry has historically contained 29 legal pairs, but the walkthrough reads the actual count from the JSON data and must continue to work when a workload or tile is added.

### Step 9 — Three-session tile performance experiment

benchmarks/benchmark_attention_sessions.py measures the standard seven-workload timing matrix. It performs ten warmups, 30 CUDA-event samples and five launches per sample. It rotates/reverses tile order across three sessions to reduce systematic order bias, and retains the per-session measurements plus the aggregate.

This is a performance experiment, not a correctness substitute. Each candidate is checked numerically before timing. Review median and P90 together; also inspect the baseline measurements and session spread before claiming a speedup. The reference baseline is the project's unfused FP32 CuPy baseline, not a vendor-optimized attention implementation.

### Step 10 — DRAM load-latency-hint experiment

benchmarks/benchmark_load_latency_hints.py evaluates six hint values (0, 2, 4, 6, 8 and 10) for two fixed workload/tile combinations over three sessions: 36 hint/session trials in total. Each variant must pass the independent NumPy tolerance before its timing is accepted. Session order is rotated to make the comparison less sensitive to run order.

A lower single-session median is not sufficient to recommend a hint. The script considers the three-session spread and only recommends a non-default hint when the observed gain clears its configured threshold. This step does not launch Nsight Compute; profiling is a separate activity.

## 4. Guided kernel tour

The guided tour shows a bounded source range from src/cutile_attention.py, explains its role in the algorithm and then runs a test that exercises the corresponding property:

1. **Input/tile contract:** validates tensor shapes, dimensions, layout and tile divisibility before launch.
2. **Specialization and launch grid:** allocates the output, specializes the kernel and maps program IDs to batch, head and query tile.
3. **Query tile and causal loop bound:** initializes the online-softmax state and bounds the key/value loop to relevant tiles.
4. **Key/value load, score and causal mask:** loads K/V blocks, calculates scaled QK scores and masks future positions with negative infinity.
5. **Online softmax and weighted values:** rescales the previous running maximum, normalization sum and weighted-value accumulator before incorporating the current block.
6. **Final normalization and output:** divides the running weighted sum by its normalization sum and writes the FP16 result.

The demonstration then reviews the unfused FP32 CuPy baseline, executes selected evidence tests, and runs every currently collected GPU pytest node separately. Parameterized tests are separate node IDs, so each workload/tile/dimension variant is visible in the output. It finally runs each legal registry pair as a separate numerical-matrix invocation. The test and legal-configuration counts are computed from current pytest/JSON data; they are not hard-coded to 48 or 29.

## 5. Reusable demo and Nsight Compute

The reusable demo combines:
1. GPU/Python preflight and cuTile smoke check.
2. Dynamic collection and execution of the complete required native GPU test suite.
3. Deterministic NPZ generation.
4. The actual cuTile command-line inference path.
5. Independent NumPy verification.
6. A presentation-sized tile sweep (unless -SkipTileSweep is supplied).
7. An optional Nsight Compute attempt (when -WithProfile is supplied).

The presentation-sized sweep compares five tile shapes across two workloads. It is intentionally smaller than the full three-session, seven-workload timing study in Step 9.

The Nsight option attempts to capture one warmed medium_d128, 64×64 kernel. A successful command must create a .ncu-repz or .ncu-rep report; the script checks that report exists before reporting NCU_STATUS=CAPTURED. If Nsight reports ERR_NVGPUCTRPERM, the script marks the capture as blocked and preserves the log rather than pretending counters were collected. That permission issue is an environment restriction, not a numerical correctness failure. The report does not claim a capture succeeded in that case.

## 6. Logs and failure handling

All walkthrough output and generated demo files are written below:

~~~text
.tmp/logs/
~~~

Files are timestamped. The standard walkthrough emits separate runtime, core-test, edge-test, input-generation, CLI, reference-check, numerical-matrix, tile-experiment and latency-hint logs. The guided tour records the one-by-one test stream and per-configuration JSON outputs. The reusable demo records test collection, test execution, CLI and optional NCU logs. The benchmark scripts also write machine-readable JSON and CSV results.

When a step fails:
1. Read the exact failing command and exit code in the terminal.
2. Open the matching timestamped log in .tmp/logs/.
3. Verify that the project-local Python and expected GPU are in use.
4. Confirm REQUIRE_CUTILE=1 was applied for required GPU tests; missing GPU dependencies must fail rather than skip.
5. For a failed numerical matrix, inspect the JSON row's case, tile, max absolute error, P99 absolute error, RMSE, tolerance envelope and error text.
6. For a timing experiment, separate numerical failures from timing instability; do not modify the default tile or hint based on a single measurement.
7. For NCU, distinguish a missing executable, kernel-launch failure and counter-permission denial.

The interactive menu catches a failed operation, prints its error and returns to the menu. The scripted walkthroughs and CI wrappers propagate non-zero exit codes so a failing stage cannot be misreported as a successful demo.

## 7. Local repository synchronization in the self-hosted validation

The self-hosted validation workflow uses actions/checkout@v4 pinned to the exact event SHA in its isolated Actions workspace, then verifies the checked-out HEAD and required source files. It separately checks the local project checkout at C:\Users\Public\GPU-cutile-project\gpu-nvidia before synchronizing it. The checkout action cleans only the Actions workspace, never the separately guarded local project root.

The local checkout is fast-forwarded only when:
- it is a Git checkout of xemptynothing/gpu-nvidia,
- it is on main,
- its working tree is clean, and
- local main is strictly behind or equal to origin/main.

The workflow does not use reset --hard or git clean. Dirty or diverged local state is preserved and reported as a blocker instead of being overwritten. On success, the log prints LOCAL_REPO_STATUS=SYNCED and the exact local HEAD SHA.

## 8. Verified result boundary

The last previously verified native RTX 4060 validation was GitHub Actions run [#42075683189](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42075683189), which passed the genuine cuTile smoke, 57 required GPU correctness/edge executions, three optional attention-objective tests, the legal numerical matrix, the CLI oracle check, benchmark and profiling stages.

The CPU/reference workflow [#42082132110](https://github.com/xemptynothing/gpu-nvidia/actions/runs/42082132110) passed the CPU reference suite and Python syntax checks.

Those are historical run results on the recorded commit, not proof that a later source edit has passed. The newest mentor-validation run must be checked after changing these scripts. Do not use old timing values as current measurements unless the same current commit has been measured again.
