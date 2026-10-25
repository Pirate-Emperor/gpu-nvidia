# Causal attention with NVIDIA cuTile

This project implements the forward pass of causal scaled dot-product attention as a fused cuTile kernel. It runs on the Windows RTX 4060. The A100 work in this repository is ordinary CUDA C++ and is kept as a separate comparison.

## Code

- src/cutile_attention.py — cuTile kernel and launch path.
- src/reference.py — FP32 NumPy reference.
- src/online_reference.py — tiled online-softmax reference.
- src/baseline.py — deliberately unfused FP32 CuPy baseline.
- tests/ — correctness, input-contract and regression tests.
- run_attention.py — command-line interface.
- benchmarks/ — timing and profiling tools.

## Current status

The last complete RTX 4060 run, on commit bf5a623, passed the 48 core GPU tests, 29 numerical case/tile checks, CLI verification, three timing sessions and two Nsight Compute configurations: https://github.com/xemptynothing/gpu-nvidia/actions/runs/39422402438

The kernel was compacted after that run. Treat the old measurements as historical until the current source passes a fresh native GPU run.

The timing baseline is the unfused FP32 CuPy implementation, not FlashAttention or an optimized PyTorch attention backend.
The ordinary CUDA C++ controls are not actual NVIDIA cuTile runs.

## Remaining course work

- Recheck the current kernel on the RTX 4060.
- Measure first-call/JIT time with a clean cache for the long cases.
- Complete the framework attention comparison and record the backend used.
- Use the profiler evidence to test another kernel change.
- Save readable Nsight Compute Details, Raw, occupancy, memory and warp-state captures.
- Finish the report and record each person's actual contribution.

See the [completion checklist](docs/COURSE_PROJECT_COMPLETION_CHECKLIST.md), [two-page summary](docs/PROJECT_TWO_PAGER.md), [test matrix](docs/TEST_MATRIX.md) and [full experiment report](Report.md).

## Run the core tests

Use the existing project virtual environment on a supported NVIDIA GPU with CuPy and cuTile installed.

~~~powershell
$py = ".\.venv-cutile-windows\Scripts\python.exe"
& $py scripts\cutile_smoke.py
if ($LASTEXITCODE -ne 0) { throw "cuTile runtime check failed." }

$env:REQUIRE_CUTILE = "1"
& $py -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py
$testCode = $LASTEXITCODE
Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue
if ($testCode -ne 0) { throw "GPU tests failed." }
~~~

Generate and check a CLI output:

~~~powershell
& $py scripts\generate_inputs.py --output .tmp\sample.npz --batch 1 --heads 1 --seq 64 --head-dim 32 --seed 20261024
& $py run_attention.py --input .tmp\sample.npz --output .tmp\output.npy --backend cutile --tile-q 64 --tile-k 64
& $py scripts\verify_cli_output.py --input .tmp\sample.npz --output .tmp\output.npy
~~~

The runtime check confirms compilation and launch. The 48 tests cover attention correctness; the 29-case numerical matrix is a separate check.

## Project tracking files

- docs/TODO_CHECKLIST.md
- docs/README.md
- docs/EXPERIMENT_REPORT.md
- docs/CUTILE_RTX4060_RESULTS_2026-10-19.md
- docs/TESTING_CONTRACT.md
- docs/TEST_MATRIX.md
- docs/PROGRESS.md
- results/STATUS.md
