# Testing contract

The GPU kernel is checked against src/reference.py, an independent FP32 NumPy implementation. The reference does not call the cuTile kernel.

## Input contract

- Q, K and V are contiguous FP16 arrays with shape [B,H,N,D].
- Supported head dimensions are 32, 64 and 128.
- The required path uses causal self-attention and Dv = D.
- Sequence lengths must fit the selected query and key tile sizes.
- NaN, infinity, unsupported dimensions and invalid layouts must be rejected.

## Numerical check

For each output element:

abs(actual - reference) <= 0.02 + 0.02 * abs(reference)

The suite also checks output shape, dtype and finite values. Do not widen the tolerance just to accept a faster kernel.

## What is tested

- 29 legal workload/tile combinations.
- One additional tile configuration.
- Small and large input magnitudes.
- Constant-value and zero-score properties.
- Future-key and future-value invariance.
- Invalid shapes, dimensions, values and memory layouts.

The core GPU suite contains 48 executions. The separate 29-row numerical matrix is not part of that count.

## Run the GPU suite

~~~powershell
$py = ".\.venv-cutile-windows\Scripts\python.exe"
$env:REQUIRE_CUTILE = "1"
& $py -m pytest -v -ra tests\test_gpu_correctness.py tests\test_gpu_edge_cases.py
$testCode = $LASTEXITCODE
Remove-Item Env:REQUIRE_CUTILE -ErrorAction SilentlyContinue
if ($testCode -ne 0) { throw "GPU tests failed." }
~~~

The REQUIRE_CUTILE flag makes missing GPU dependencies a failure, not a skip.

Every kernel change must pass the full suite before timing or profiling. Keep failed tests as regression tests.
