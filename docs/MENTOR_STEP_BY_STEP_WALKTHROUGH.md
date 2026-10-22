# Mentor Walkthrough — Run Each Test and Experiment in Order

**Target:** native Windows RTX 4060 running the project-local cuTile venv.  
**Purpose of this session:** show the 48 required GPU test executions, independently check legal tile configurations, inspect the unfused baseline and fused cuTile source, and reproduce the tile/hint experiments. **Nsight Compute is deliberately left for the separate profiler session.**

## Start here

From PowerShell in the repository root:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1
```

The script pauses before each stage so you can read the purpose and command, press Enter, and watch that stage execute. The verbose pytest output shows the individual tests rather than only one aggregate PASS. It saves separate timestamped logs and result JSON/CSV files under `.tmp\logs`.

The default walkthrough runs the full 29-pair numerical matrix, the full seven-workload/three-session tile experiment and the three-session latency-hint experiment. These experiments are longer than the core tests. To focus on the correctness + tile experiment while leaving the separate hint sweep for another time:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -SkipLatencyHintExperiment
```

`-NoPause` removes the prompts; `-SkipNumericalMatrix`, `-SkipTileExperiment` and `-SkipLatencyHintExperiment` each skip only that explicitly named stage. Skipping a stage does not turn its evidence green. The script has no Nsight Compute step.

## Stage-by-stage commands and what each one proves

Run from the repository root. Set the project Python executable once:

```powershell
$py = ".\.venv-cutile-windows\Scripts\python.exe"
$logDir = ".\.tmp\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
```

If the venv is not under the current checkout, use the path the existing `mentor_demo.ps1` reports as `PYTHON_EXE`. Do not replace it with a system Python just to get the command running.

### Step 1 — record the environment

```powershell
& $py --version
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv
```

**Why:** runtime correctness/performance claims only apply to the machine, driver and Python environment that actually executed them. This project’s actual-cuTile evidence is from the native Windows RTX 4060 (SM 8.9, driver 616.92), not the Linux A100/R550 ordinary-CUDA control.

**What to show:** GPU name, driver, compute capability and memory. Check that the active interpreter is `.venv-cutile-windows`.

### Step 2 — genuine cuTile smoke test

```powershell
& $py .\scripts\cutile_smoke.py
```

**Why:** before debugging attention, prove the cuTile compiler/runtime can compile, launch and synchronize a real cuTile kernel on this GPU. This is a tiny vector-add smoke, not evidence of attention correctness or performance.

**Recorded result:** PASS; 4,096 outputs were checked exactly and stream synchronization passed. The script prints `CUTILE_SMOKE=PASS`.

### Step 3 — run the 34 correctness-file executions by themselves

```powershell
$env:REQUIRE_CUTILE = "1"
& $py -m pytest -v -ra .\tests\test_gpu_correctness.py
```

**Why:** `-v` prints each parameterized test ID and result. `REQUIRE_CUTILE=1` ensures missing CuPy/cuTile is a failure rather than a silently skipped GPU test. This file covers the 29 main legal case/tile combinations, one extra asymmetric tile, small/large logits, causal future-value invariance and rejection of a non-divisible sequence.

The successful recorded suite had **34 passed** in this file. If this stage fails, stop and inspect the first failing assertion and the test log before doing performance work.

### Step 4 — run the 14 edge-case executions by themselves

```powershell
& $py -m pytest -v -ra .\tests\test_gpu_edge_cases.py
Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue
```

**Why:** these tests make the algorithm prove invariants, not merely produce plausible random-number outputs. They check constant (V), zero logits/prefix averages, tile-to-tile agreement, causal effects of future (K/V), rejection of NaN/Inf and non-contiguous input, unsupported/mismatched dimensions and non-divisible tile shapes.

The successful recorded suite had **14 passed** in this file. Together, Steps 3 and 4 make up the **48/48 required GPU test executions**. These are parameterized test executions, not 48 distinct Python functions.

### Step 5 — generate the same input on every rerun

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$inputFile = Join-Path $logDir "mentor_input_$stamp.npz"
$outputFile = Join-Path $logDir "mentor_output_$stamp.npy"
& $py .\scripts\generate_inputs.py --output $inputFile --batch 1 --heads 1 --seq 64 --head-dim 32 --seed 20261018
```

**Why:** the fixed seed and saved input file make the CLI result auditable. It is important that the independent reference and the kernel use the exact same Q/K/V data.

### Step 6 — run the course-facing command-line interface

```powershell
& $py .\run_attention.py --input $inputFile --output $outputFile --backend cutile --tile-q 64 --tile-k 64
```

**Why:** this proves the delivered interface can load the course-format NPZ input, call the cuTile implementation and write a real NumPy output file. It covers more than directly calling a unit-test helper.

**Expected print:** backend, Q/K/V shapes, output shape and output dtype. The recorded output was FP16, finite and correctly shaped.

### Step 7 — use an independent NumPy oracle to check the saved file

```powershell
& $py .\scripts\verify_cli_output.py --input $inputFile --output $outputFile
```

**Why:** verify the file written by the CLI after it has been saved. The check enforces shape, FP16 dtype, finite values and
[
|O_{actual}-O_{reference}|\le 0.02+0.02|O_{reference}|.
]

The expected answer comes from the independent full-matrix FP32 NumPy reference. The cuTile kernel is never used to manufacture its own expected result. Recorded marker: `CLI_REFERENCE_CHECK=PASS`.

### Step 8 — run the full legal numerical matrix (29 rows)

```powershell
$matrixFile = Join-Path $logDir "mentor_legal_numerical_matrix_$stamp.json"
& $py .\scripts\mentor_numerical_matrix.py --output $matrixFile
```

This dedicated matrix script is added for the step-by-step presentation. It runs each **legal** mandatory workload/tile pair, prints a PASS/FAIL line with absolute error, P99 absolute error, RMSE and tolerance-envelope usage, and writes JSON plus a sibling CSV. The successful historical project run reported 29/29 and zero violations; this newly added explicit script must produce its own fresh output before it is claimed as freshly validated.

**Why the legality filter exists:** the core fixed-tile contract supports positive power-of-two query/key tiles and requires the sequence length (N) to be divisible by both (TQ) and (TK). It also requires finite contiguous FP16 input, (D\in\{32,64,128\}), (Dv=D), matching Q/K/V batch/sequence dimensions and matching Q/K head dimensions. The matrix uses the same seven frozen workloads and the five main tile candidates. Illegal pairs are not counted as successes; exactly 29 pairs are legal.

### Step 9 — run the full tile benchmark in three order-varied sessions

```powershell
$tileFile = Join-Path $logDir "mentor_tile_experiment_$stamp.json"
& $py .\benchmarks\benchmark_attention_sessions.py --output $tileFile --warmup 10 --iterations 30 --repeats-per-sample 5
```

This session driver runs the existing benchmark harness three times over all seven mandatory cases and up to five tile configurations. It rotates/reverses the tile order to reduce the risk that a fixed position in the sweep determines the result. It retains three raw session JSON/CSV pairs plus an aggregate JSON/CSV with one row per legal case/tile pair.

**Why every timing parameter exists**

| Parameter / field | Setting | Purpose |
|---|---:|---|
| `case` | 7 frozen inputs | Fixes (B,H,N,D,Dv) and random seed. Each tile is compared on the same workload instead of mixing shape changes with tile changes. |
| `tiles` | 32×64, 64×64, 64×128, 128×64, 128×128 | Tests different query/key block shapes. A larger tile may reduce loop overhead but increase live data/register/shared-memory pressure; it is a hypothesis, not a guarantee of speed. |
| `warmup` | 10 calls per configuration | Runs the path before samples are recorded, reducing one-off startup/scheduling effects. It does not turn a first-call measurement into a clean-cache compilation measurement. |
| `iterations` | 30 CUDA-event samples | Produces a latency distribution rather than a single timing. Median is the main comparison; P90 describes the slower tail. The benchmark rejects fewer than 30 iterations. |
| `repeats-per-sample` | 5 launches | One CUDA event interval encloses five identical launches; elapsed time is divided by five. This improves timer granularity for short kernels. |
| Timed launches | 30×5 = 150 per configuration/session | These are the launches inside the measured samples; 10 warmups are additional and excluded. Three sessions therefore supply 450 timed launches per legal configuration. |
| `output` | JSON and CSV | Preserves environment and source metadata, shape/seed, each session median/P90, baseline-before/after, drift, speedup and separate first-call wall time. This is the evidence to inspect after the console table. |
| `baseline` | unfused FP32 CuPy | Gives an interpretable performance control. It materializes the score/probability matrices and is deliberately not a vendor-optimized attention kernel. |
| Baseline bracketing | before and after tile sweep | Detects clock/background-state drift during a session. Treat a small tile win as inconclusive if baseline drift is large. |
| First-call time | reported separately | Includes compile/runtime/cache effects and may hit an existing compiler cache. It is not substituted for steady-state latency or called a clean JIT measurement. |

The event benchmark measures device-resident functions. The input copies are outside the timed calls. The benchmark also reports output-byte throughput, which is **output tensor bytes only**, not total DRAM bandwidth.

### Step 10 — test the cuTile DRAM load-latency hint

```powershell
$hintFile = Join-Path $logDir "mentor_latency_hints_$stamp.json"
& $py .\benchmarks\benchmark_load_latency_hints.py --output $hintFile --warmup 10 --iterations 30 --repeats-per-sample 5 --sessions 3
```

**Why this experiment exists:** it holds the workload and tile shape fixed while varying only the cuTile load-latency hint. It first compares each candidate output with the independent FP32 NumPy reference; only correct variants are timed. Hints are `0, 2, 4, 6, 8, 10`. It tests medium D=128 / tile 64×64 and longer-batched D=128 / tile 64×128, six hints × two workloads × three sessions = **36 session trials**.

A candidate is recommended only if its median gain clears both a 1% minimum and half the observed baseline/candidate spread. The script keeps compiler-inferred hint `0` as the default if a candidate does not pass that noise-aware rule. The saved result contains recommendations and raw trials.

**This stage does not run Nsight Compute.** Profiler interpretation is intentionally reserved for the later session.

## What the recorded experiments showed

These are the stored results from the completed three-session experiment, not numbers prefilled into a live run. Your fresh output may differ because of GPU clocks, temperature, desktop load and compilation-cache state.

### Tile sweep — medium and longer-batched D=128

| Case | Tile | Median ms | P90 ms | Speedup vs unfused FP32 CuPy |
|---|---:|---:|---:|---:|
| Medium D=128 | 32×64 | 0.069632 | 0.070659 | 5.62× |
| Medium D=128 | 64×64 | 0.058368 | 0.058814 | 6.70× |
| Medium D=128 | 64×128 | 0.058368 | 0.059044 | 6.70× |
| Medium D=128 | 128×64 | 0.079053 | 0.079258 | 4.94× |
| Medium D=128 | 128×128 | 0.107315 | 0.107950 | 3.64× |
| Longer batched D=128 | 32×64 | 1.142989 | 1.146900 | 14.97× |
| Longer batched D=128 | 64×64 | 0.932966 | 0.936468 | 18.33× |
| Longer batched D=128 | 64×128 | 0.915558 | 0.921457 | 18.69× |
| Longer batched D=128 | 128×64 | 1.017139 | 1.020314 | 16.83× |
| Longer batched D=128 | 128×128 | 1.285120 | 1.289237 | 13.32× |

**Decision:** keep 64×64 as the general default. 64×128 was about 1.86% faster on the longer-batched case but not on the long unbatched case. Medium D=128 medians tied; baseline drift there reached 10.83%, so that part of the tile timing table is not decisive. The longest workloads are a cleaner performance story than the shortest workloads, where Windows scheduling noise was high.

### Load-latency hint results

| Workload | Hint 0 median | Best candidate | Gain vs 0 | Decision |
|---|---:|---:|---:|---|
| Medium D=128 / 64×64 | 0.058342 ms | Hint 10: 0.053248 ms | 8.73% | Strong timing candidate, matched profiler review pending |
| Longer batched D=128 / 64×128 | 0.922010 ms | Hint 2: ~0.920576 ms | 0.16% | Keep hint 0; difference is below observed variation |

All 36 historical hint trials passed the NumPy mixed-tolerance check. Keep global `LOAD_LATENCY=0`; hint 10 is not promoted as a global/default setting until the matched profile review is complete.

### Full correctness and numerical summary

| Check | Recorded result |
|---|---|
| Genuine cuTile smoke | PASS; 4,096 outputs exactly checked |
| Required attention suite | 48 passed, 0 failed |
| Independent numerical matrix | 29 legal case/tile pairs; zero tolerance violations |
| Maximum absolute error | 0.0009765625 |
| Largest per-configuration P99 absolute error | 0.00048828125 |
| Largest per-configuration RMSE | 0.000086860741 |
| CLI output shape/dtype/finiteness and independent NumPy oracle | PASS |

The largest tolerance-envelope ratio was 2.4367% of the allowed envelope; that is not a 2.4367% output error.

### Historical CUDA control ladder (separate from actual cuTile)

The earlier v1/v2/v3/current and v4/v5 ladder used ordinary CUDA C++ control kernels on the A100 path. It is useful to show why optimization hypotheses were accepted or rejected, but those measurements are **not actual cuTile results and should not be ranked against RTX 4060 numbers above**.

Best median per approach in one earlier ladder run (ms):

| Workload | v1 | v2 | v3 | Current CUDA control |
|---|---:|---:|---:|---:|
| short D=32 | 0.111 | 0.066 | 0.055 | 0.055 |
| short D=64 | 0.119 | 0.073 | 0.060 | 0.061 |
| short multi-head D=64 | 0.385 | 0.102 | 0.077 | 0.081 |
| medium D=64 | 1.727 | 0.234 | 0.180 | 0.193 |
| medium D=128 | — | 0.947 | 0.815 | 0.792 |
| long D=128 | — | 7.188 | 6.556 | 6.131 |
| longer batched D=128 | — | 22.895 | 26.533 | 25.678 |

The same run measured first-compile wall times of v1 1652 ms, v2 717 ms, v3 1063 ms and current 808 ms. These are compile/JIT-associated times, not steady-state kernel medians. The absolute results varied across workflow runs, so treat them as historical engineering measurements.

v4 (persistent Q fragments in registers) passed 30 correctness tests but had mixed timing; the PTXAS summary showed no resource improvement over its CUDA control predecessor (32 registers, 33,792 bytes shared memory, zero spills; equal 15,136-byte cubin for current/v4). v5 (half2-oriented Q/K loads) passed 30/30 tests, but its gains on long and longer-batched cases were only about 0.3–0.6%, with mixed short-case results. Neither variant was promoted. The detailed row-by-row v5 comparison is in [the experiment report](EXPERIMENT_REPORT.md) and the algorithm/source context is in [the kernel walkthrough](CUTILE_KERNEL_WALKTHROUGH.md).

## Experiments and paths not to overclaim

- **128×128 tile:** it was slower on the long D=128 workloads, so it was not promoted.
- **Shared-tile v4 and v5 prototypes:** historical ordinary-CUDA/A100 control experiments, not current cuTile implementation evidence. Their correctness tests passed in their own workflows but performance was mixed, so neither prototype was promoted.
- **PyTorch SDPA:** comparison is blocked before the cuTile bridge by `CUBLAS_STATUS_INVALID_VALUE` in a plain `torch.bmm` on the pinned CUDA 13.0 wheel. No PyTorch speedup or successful gradient result is claimed.
- **Clean-cache JIT:** a controlled cold-cache measurement for the longest specializations remains open. First-call fields may be compiler-cache hits.
- **Nsight Compute:** intentionally deferred. No profiler screenshot, occupancy metric or NCU heuristic is required in today's demonstration.

## Source code to open during the walkthrough

```powershell
Get-Content .\src\baseline.py
Get-Content .\src\cutile_attention.py
Get-Content .\src\reference.py
Get-Content .\src\contract.py
Get-Content .\tests\test_gpu_correctness.py
Get-Content .\tests\test_gpu_edge_cases.py
Get-Content .\benchmarks\benchmark_attention.py
Get-Content .\benchmarks\benchmark_load_latency_hints.py
```

Use [the code walkthrough](CUTILE_KERNEL_WALKTHROUGH.md) for the line-by-line algorithm explanation, baseline design, test rationale and mathematical online-softmax recurrence.

## Speaking order

1. Show the GPU and cuTile smoke, then run each pytest file separately. Explain why missing cuTile is treated as a failure.
2. Open baseline.py, then cutile_attention.py, then reference.py. Explain the causal mask, the online-softmax state and why the independent oracle is separate.
3. Show the 29 legal numerical rows and the saved CLI oracle result.
4. Show the three-session tile table. Compare median, P90 and baseline drift, not just the lowest median.
5. Show the latency hint experiment's accuracy gate and noise-aware recommendation. Explain why hint 0 remains the global default.
6. Stop before Nsight Compute; profiler interpretation is for the later session.
