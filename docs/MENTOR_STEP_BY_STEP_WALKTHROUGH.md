# Mentor walkthrough

Target: native Windows RTX 4060 with the project cuTile environment.

## Start the menu

From the repository root:

~~~powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1
~~~

Choose a mode from the menu. Guided mode explains the main kernel sections, runs related tests, then executes each of the 48 GPU tests separately. It also runs the legal numerical configurations listed in tests/legal_attention_configs.json.

To start guided mode directly:

~~~powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Guided
~~~

To skip the longer timing experiments:

~~~powershell
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\mentor_step_by_step.ps1 -Guided -SkipTileExperiment -SkipLatencyHintExperiment
~~~

## What each stage checks

1. Environment — records Python, GPU and installed package versions.
2. cuTile runtime check — compiles and launches a small real cuTile kernel.
3. Core suite — runs 34 correctness tests.
4. Edge suite — runs 14 edge-case tests.
5. CLI — generates a fixed input file and runs attention.
6. Output check — compares the saved result against the independent NumPy reference.
7. Numerical matrix — checks each legal case/tile pair in the JSON registry.
8. Tile timing — compares the fixed seven-workload matrix over three order-rotated sessions.
9. Load-latency hint — measures candidate settings; a faster median alone does not justify changing the default.

The standard timing matrix is fixed. Adding a case to the JSON file expands the numerical matrix but does not silently add a new benchmark workload.

## Current result boundary

The last complete native GPU run is #39422402438. The kernel was compacted afterward, so re-run the runtime check, 48 tests and CLI check on current main before using the historical timing numbers as current results.

The benchmark baseline is unfused FP32 CuPy, not vendor-optimised attention. Nsight Compute is a separate session; use the [profiler analysis](CUTILE_NCU_PROFILE_ANALYSIS_2026-10-19.md) and [test matrix](TEST_MATRIX.md) for the saved results.
