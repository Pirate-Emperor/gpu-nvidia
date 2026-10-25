# Run cuTile on the Windows RTX 4060

The actual cuTile target is the local Windows RTX 4060 (compute capability 8.9, driver 616.92). The Linux A100 with driver R550.90.07 is used for ordinary CUDA C++ comparisons; do not treat those runs as cuTile results.

## Before running

Open PowerShell at the repository root and check the environment:

~~~powershell
nvidia-smi
.\.venv-cutile-windows\Scripts\python.exe --version
.\.venv-cutile-windows\Scripts\python.exe -c "import numpy, cupy, cuda.tile; print(numpy.__version__, cupy.__version__)"
~~~

The recorded environment used Python 3.12.10, NumPy 2.5.4, CuPy 14.2.0 and cuda-tile 1.6.0. Check the versions on the machine before relying on an old report.

## Run the checks

~~~powershell
$py = ".\.venv-cutile-windows\Scripts\python.exe"

& $py scripts\cutile_smoke.py
if ($LASTEXITCODE -ne 0) { throw "cuTile runtime check failed." }

$env:REQUIRE_CUTILE = "1"
& $py -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py
$testCode = $LASTEXITCODE
Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue
if ($testCode -ne 0) { throw "GPU correctness tests failed." }

& $py scripts\generate_inputs.py --output .tmp\sample.npz --batch 1 --heads 1 --seq 64 --head-dim 32 --seed 20261024
if ($LASTEXITCODE -ne 0) { throw "Input generation failed." }

& $py run_attention.py --input .tmp\sample.npz --output .tmp\output.npy --backend cutile --tile-q 64 --tile-k 64
if ($LASTEXITCODE -ne 0) { throw "cuTile CLI failed." }

& $py scripts\verify_cli_output.py --input .tmp\sample.npz --output .tmp\output.npy
if ($LASTEXITCODE -ne 0) { throw "CLI output check failed." }
~~~

The runtime check tests compilation and launch. The 48-test suite checks attention behaviour. The CLI check compares the saved output with the independent NumPy calculation.

## Nsight Compute

After correctness passes, run the profiler workflow or use benchmarks/profile_fused.py with ncu. Save the original report files, metric exports and GUI captures. Include Summary, Details, Raw, Occupancy, Memory, Scheduler and Warp State views when available.

Use profiler counters to form a hypothesis, then test it with a code change, full correctness rerun and repeated unprofiled timing. NCU measurements include profiler overhead and should not replace normal timing results.

## Current status

The last full native run is #39422402438 on commit bf5a623. The kernel changed after that run, so run the checks above on current main before citing the older results as current.
