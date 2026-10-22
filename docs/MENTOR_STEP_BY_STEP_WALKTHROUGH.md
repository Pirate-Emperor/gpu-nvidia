# Mentor Walkthrough — Run Each Test and Experiment in Order

**Target:** native Windows RTX 4060 running the project-local cuTile venv.  
**Purpose of this session:** show the 48 required GPU test executions, independently check legal tile configurations, inspect the unfused baseline and fused cuTile source, and reproduce the tile/hint experiments. **Nsight Compute is deliberately left for the separate profiler session.**

## Start here

From PowerShell in the repository root:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1
```

The script opens an interactive menu. It has separate options for the standard demonstration, code-by-code explanation with tests, all 48 required tests, one selected required test, one numerical configuration, all configured numerical configurations, editing and validating the configuration file, and the two performance experiments. Test and result files are written under `.tmp\logs`.

Choose **2** in the menu for guided source-and-test mode. It displays sections of `src/cutile_attention.py`, explains what each section does, then runs a related test. It subsequently displays and runs each of the 48 required test executions separately. Finally, it runs every legal numerical configuration registered in the JSON file, one at a time. If additional workloads or tile candidates have been registered, the guided numerical section includes them automatically.

To start guided mode directly without opening the menu:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Guided
```

To omit the longer performance experiments in guided mode:

```powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Guided -SkipTileExperiment -SkipLatencyHintExperiment
```

The standard timing experiment still measures the seven fixed benchmark workloads and five timing tile choices. The editable JSON list expands numerical accuracy checks; adding a workload does not automatically include it in the performance timing experiment. Neither mode runs Nsight Compute.

## Extend the legal numerical test list

The file `tests/legal_attention_configs.json` is the editable list used by the numerical validation matrix. The initial file contains seven workloads and five tile candidates, which produce 29 supported workload/tile combinations.

Use main-menu option **7** to add a workload or a tile candidate. The interactive editor checks each entry before saving:
- Tile dimensions must be positive powers of two.
- The head dimension must be 32, 64 or 128, and value dimension must equal the head dimension.
- Batch size, head count and sequence length must be positive integers.
- A workload must have at least one tile for which the sequence length divides evenly by both tile dimensions.
- A tile must apply to at least one registered workload.
- Duplicate names and duplicate tile pairs are rejected.

Invalid entries produce a warning and prompt for a correction; they are not saved. Large workloads also generate a memory warning because the independent full-matrix NumPy reference can require substantial memory. Main-menu option **8** checks and lists the current file; option **9** opens it in Notepad. Validate the file after manual edits.

Choose main-menu option **6** to test every legal combination in the file, including additional entries, and option **5** to run one pair or all tiles for a chosen workload. The fixed 48-test suite remains separate and retains its original count. The numerical matrix may exceed 29 rows when the registry is extended.

A direct list/validation command is:

```powershell
$py = ".\.venv-cutile-windows\Scripts\python.exe"
& $py .\scripts\legal_config_registry.py list
& $py .\scripts\legal_config_registry.py validate
```

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

### Step 8 — run every legal numerical configuration in the JSON registry

```powershell
$matrixFile = Join-Path $logDir "mentor_legal_numerical_matrix_$stamp.json"
& $py .\scripts\mentor_numerical_matrix.py --output $matrixFile
```

This dedicated matrix script reads `tests/legal_attention_configs.json`. It runs each **legal** registered workload/tile pair, prints a PASS/FAIL line with absolute error, P99 absolute error, RMSE and tolerance-envelope usage, and writes JSON plus a sibling CSV. The initial list reports 29/29 pairs; adding a valid workload or tile increases the number automatically. The script checks the configuration file before using it.

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

### What the three benchmark sessions mean

The three sessions are three separate runs of the benchmark program, not three samples and not three different input cases. Each session repeats the same seven workloads and the same five candidate tile shapes wherever each shape is supported. Each session uses the same fixed input seeds and the same timing settings.

The driver changes the order of tile shapes to reduce a systematic order effect:

| Session | Tile order |
|---|---|
| 1 | 32×64 → 64×64 → 64×128 → 128×64 → 128×128 |
| 2 | 64×64 → 64×128 → 128×64 → 128×128 → 32×64 |
| 3 | 128×128 → 128×64 → 64×128 → 64×64 → 32×64 |

Why change the order? The GPU may warm up, change clock speed or temperature, or experience background activity as the run proceeds. If a configuration is always measured first, it may be measured under different conditions from one always measured last. Rotating and reversing the order helps reveal that risk. Three sessions do not eliminate all measurement noise, but they show whether a claimed win repeats across separate runs.

For each case/tile pair, each session performs 10 warmup calls, then 30 timing samples of five launches each: **150 timed launches per session**. Across three sessions, that is **450 timed launches and 30 warmup calls per configuration**. The aggregate retains the three session medians and P90s, reports the median across those session summaries, and calculates the spread between the slowest and fastest session median.

### The seven workloads behind the 29 legal configurations

In the shape notation `(B,H,N,D)`, `B` is batch size, `H` is attention-head count, `N` is sequence length, and `D` is the dimension of each query/key/value vector.

| Case | Shape (B,H,N,D) | Legal tiles | Count |
|---|---:|---|---:|
| `short_d32` | (1,1,64,32) | 32×64, 64×64 | 2 |
| `short_d64` | (1,1,64,64) | 32×64, 64×64 | 2 |
| `short_multihead_d64` | (1,4,128,64) | all five candidates | 5 |
| `medium_d64` | (1,8,256,64) | all five candidates | 5 |
| `medium_d128` | (1,8,512,128) | all five candidates | 5 |
| `long_d128` | (1,16,1024,128) | all five candidates | 5 |
| `longer_batched_d128` | (2,8,2048,128) | all five candidates | 5 |
| **Total** | **7 workloads** | | **29** |

The five tile candidates are 32×64, 64×64, 64×128, 128×64 and 128×128. The two shortest cases have sequence length 64, so key tiles of 128 and query tiles of 128 do not satisfy the fixed-divisibility requirement. The other cases have sequence lengths that divide evenly by all five candidates.

### Why compare medium D=128 with longer-batched D=128?

Both workloads use vector dimension `D=128` and eight attention heads, so the data width and head count stay fixed. The important differences are batch size and sequence length:

| Property | Medium D=128 | Longer-batched D=128 |
|---|---:|---:|
| Batch size (`B`) | 1 | 2 |
| Heads (`H`) | 8 | 8 |
| Sequence length (`N`) | 512 | 2,048 |
| Head dimension (`D`) | 128 | 128 |
| Total query positions (`B × H × N`) | 4,096 | 32,768 |
| Purpose | Mid-sized workload | Longer sequence with a larger batch |

Yes, the longer-batched experiment uses a larger batch: **2 instead of 1**. Its sequence is also four times longer. The total number of query positions is therefore eight times larger. Causal attention compares each query with its allowed prefix of keys, so the total number of score entries grows roughly with the square of sequence length. With these batch sizes, the longer-batched case processes roughly 32 times as many causal query-key pairs as the medium case. This is why its absolute latency is much larger and why a tile choice that works well for one shape may behave differently for the other.

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

### Full tile sweep — all 29 legal case/tile results

The following is the recorded three-session aggregate from the completed native RTX 4060 experiment. It is historical evidence, not output generated by filling in a new run. The new session driver will print its own measured values and save three raw session files plus an aggregate CSV/JSON. All latency values are milliseconds. Range is the min-to-max spread of the three session medians divided by their aggregate median. Baseline drift is the largest difference between the unfused baseline measured before versus after the sweep.

| Workload | Shape (B,H,N,D) | Tile | Median | P90 | Baseline | Speedup | Session range | Max baseline drift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Short D=32 | (1,1,64,32) | 32×64 | 0.015770 | 0.018473 | 0.297270 | 19.10× | 2.09% | 77.46% |
| Short D=32 | (1,1,64,32) | 64×64 | 0.015974 | 0.018125 | 0.297270 | 18.73× | 0.64% | 77.46% |
| Short D=64 | (1,1,64,64) | 32×64 | 0.016074 | 0.018186 | 0.296243 | 18.25× | 5.20% | 1.69% |
| Short D=64 | (1,1,64,64) | 64×64 | 0.016282 | 0.018862 | 0.296243 | 18.33× | 1.36% | 1.69% |
| Short multi-head D=64 | (1,4,128,64) | 32×64 | 0.016074 | 0.018810 | 0.280320 | 16.96× | 3.70% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 64×64 | 0.016179 | 0.018452 | 0.280320 | 17.15× | 1.29% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 64×128 | 0.016566 | 0.019313 | 0.280320 | 16.96× | 3.71% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 128×64 | 0.016794 | 0.018043 | 0.280320 | 16.69× | 3.66% | 4.93% |
| Short multi-head D=64 | (1,4,128,64) | 128×128 | 0.018842 | 0.020910 | 0.280320 | 14.88× | 4.35% | 4.93% |
| Medium D=64 | (1,8,256,64) | 32×64 | 0.020070 | 0.021238 | 0.269562 | 14.00× | 95.92% | 1.32% |
| Medium D=64 | (1,8,256,64) | 64×64 | 0.017171 | 0.017833 | 0.269562 | 16.25× | 109.73% | 1.32% |
| Medium D=64 | (1,8,256,64) | 64×128 | 0.019456 | 0.020480 | 0.269562 | 14.40× | 90.05% | 1.32% |
| Medium D=64 | (1,8,256,64) | 128×64 | 0.019430 | 0.022733 | 0.269562 | 14.22× | 120.16% | 1.32% |
| Medium D=64 | (1,8,256,64) | 128×128 | 0.024944 | 0.026010 | 0.269562 | 11.15× | 76.36% | 1.32% |
| Medium D=128 | (1,8,512,128) | 32×64 | 0.069632 | 0.070659 | 0.390861 | 5.62× | 5.00% | 10.83% |
| Medium D=128 | (1,8,512,128) | 64×64 | 0.058368 | 0.058814 | 0.390861 | 6.70× | 9.82% | 10.83% |
| Medium D=128 | (1,8,512,128) | 64×128 | 0.058368 | 0.059044 | 0.390861 | 6.70× | 2.46% | 10.83% |
| Medium D=128 | (1,8,512,128) | 128×64 | 0.079053 | 0.079258 | 0.390861 | 4.94× | 5.31% | 10.83% |
| Medium D=128 | (1,8,512,128) | 128×128 | 0.107315 | 0.107950 | 0.390861 | 3.64× | 3.05% | 10.83% |
| Long D=128 | (1,16,1024,128) | 32×64 | 0.334950 | 0.337736 | 4.390656 | 13.11× | 0.06% | 0.03% |
| Long D=128 | (1,16,1024,128) | 64×64 | 0.267059 | 0.269128 | 4.390656 | 16.49× | 0.92% | 0.03% |
| Long D=128 | (1,16,1024,128) | 64×128 | 0.270848 | 0.273449 | 4.390656 | 16.19× | 1.13% | 0.03% |
| Long D=128 | (1,16,1024,128) | 128×64 | 0.316109 | 0.319324 | 4.390656 | 13.89× | 0.78% | 0.03% |
| Long D=128 | (1,16,1024,128) | 128×128 | 0.411853 | 0.415846 | 4.390656 | 10.66× | 0.79% | 0.03% |
| Longer batched D=128 | (2,8,2048,128) | 32×64 | 1.142989 | 1.146900 | 17.104077 | 14.97× | 0.31% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 64×64 | 0.932966 | 0.936468 | 17.104077 | 18.33× | 0.33% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 64×128 | 0.915558 | 0.921457 | 17.104077 | 18.69× | 0.50% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 128×64 | 1.017139 | 1.020314 | 17.104077 | 16.83× | 0.23% | 0.08% |
| Longer batched D=128 | (2,8,2048,128) | 128×128 | 1.285120 | 1.289237 | 17.104077 | 13.32× | 0.32% | 0.08% |

**What the results support:**

- Keep 64×64 as the general default. It is the best recorded tile for long unbatched D=128 at 0.267059 ms (16.49× versus the unfused baseline).
- On longer batched D=128, 64×128 is the best recorded candidate at 0.915558 ms; it is about 1.86% faster than 64×64. Treat it as workload-specific, not a universal default.
- For medium D=128, 64×64 and 64×128 tie at the displayed median. The baseline drift was 10.83%, so the tile table is not decisive enough to claim a robust difference.
- Medium D=64 has extremely high session range in this Windows measurement, and short D=32 has 77.46% max baseline drift. Do not use the large speedup ratios for those noisy cases as headline claims.
- 128×128 loses on both long D=128 workloads. Larger tiles can increase resource pressure and do not guarantee lower latency.

The speedups compare with the deliberately unfused FP32 CuPy implementation only. They are not comparisons with FlashAttention or optimized PyTorch SDPA.

### Load-latency hint results — all 12 candidates

The hint experiment checks six values across two workload/tile combinations and three sessions. The session medians below are recorded results from the completed experiment; all 36 trials passed the independent NumPy mixed-tolerance check before timing.

| Workload | Hint | Session medians (ms) | Median (ms) | Gain vs hint 0 | Session range | Decision |
|---|---:|---|---:|---:|---:|---|
| Medium D=128 / 64×64 | 0 | 0.058061, 0.058342, 0.058368 | 0.058342 | baseline | 0.53% | Default |
| Medium D=128 / 64×64 | 2 | 0.055296, 0.054886, 0.054682 | 0.054886 | 5.92% | 1.12% | Faster candidate; less gain than hint 10 |
| Medium D=128 / 64×64 | 4 | 0.058368, 0.058368, 0.058368 | 0.058368 | -0.04% | 0.00% | No gain |
| Medium D=128 / 64×64 | 6 | 0.058368, 0.058448, 0.058368 | 0.058368 | -0.04% | 0.14% | No gain |
| Medium D=128 / 64×64 | 8 | 0.056525, 0.056397, 0.056515 | 0.056515 | 3.13% | 0.23% | Faster candidate; less gain than hint 10 |
| Medium D=128 / 64×64 | 10 | 0.053325, 0.053248, 0.053043 | 0.053248 | 8.73% | 0.53% | Strong timing candidate; matched profile review pending |
| Longer batched D=128 / 64×128 | 0 | 0.992256, 0.920883, 0.922010 | 0.922010 | baseline | 7.74% | Retain default |
| Longer batched D=128 / 64×128 | 2 | 0.957542, 0.920576, 0.920576 | 0.920576 | 0.16% | 4.02% | Below noise-aware threshold |
| Longer batched D=128 / 64×128 | 4 | 0.922214, 0.920883, 0.920576 | 0.920883 | 0.12% | 0.18% | Below required 5.88% gain |
| Longer batched D=128 / 64×128 | 6 | 0.921498, 0.920576, 0.920678 | 0.920678 | 0.14% | 0.10% | Below required 5.88% gain |
| Longer batched D=128 / 64×128 | 8 | 0.922214, 0.922829, 0.921600 | 0.922214 | -0.02% | 0.13% | No gain |
| Longer batched D=128 / 64×128 | 10 | 0.921293, 0.922010, 0.921088 | 0.921293 | 0.08% | 0.10% | Below required 5.88% gain |

**Decision:** keep compiler-inferred LOAD_LATENCY=0 as the global default. Hint 10 on medium D=128 is a repeatable timing candidate, but it is not promoted until the matched default-versus-10 profile is reviewed. On longer batched D=128, candidate gains are smaller than the observed spread, so the default remains 0.

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
3. Show every legal numerical row currently registered in the JSON file and the saved CLI reference check.
4. Show the three-session tile table. Compare median, P90 and baseline drift, not just the lowest median.
5. Show the latency hint experiment's accuracy gate and noise-aware recommendation. Explain why hint 0 remains the global default.
6. Stop before Nsight Compute; profiler interpretation is for the later session.
