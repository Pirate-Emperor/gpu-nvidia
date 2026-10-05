# Correctness Testing Contract

## Status

This document is the frozen testing contract for the NVIDIA cuTile attention project.

## Oracle

The correctness oracle is the independent NumPy implementation in src/reference.py.

The oracle:
- accepts only finite FP16 contiguous Q/K/V;
- converts inputs to FP32 before arithmetic;
- explicitly computes QK^T;
- applies scale 1/sqrt(D);
- applies an explicit causal upper-triangular mask;
- performs max-shift softmax;
- computes P @ V in FP32.

The optimized cuTile kernel is never used to generate expected outputs.

## Test categories

### 1. Structural contract
Tests validate rank, dtype, contiguity, batch/head agreement, sequence agreement, and Q/K head-dimension agreement.

### 2. Exact causal invariants
These are deterministic tests. They prove:
- query row 0 cannot see later keys/values;
- changing a future V cannot change earlier outputs;
- changing a future K cannot change earlier outputs;
- changing query row j cannot change query rows < j.

### 3. Numerical stability
Large finite logits must produce finite outputs.

### 4. Determinism
The same input must produce exactly the same oracle output.

### 5. Error policy
Kernel outputs will be checked with both maximum absolute error and maximum relative error. The project proposal's recommended FP16-output starting tolerance is:

abs_error <= 2e-2 + 2e-2 * abs(reference)

We will not loosen this merely to make a failing optimization pass. Any final tolerance must be justified by measured numerical behavior.

## Planned GPU/kernel test categories

Once run_attention.py exists, every implementation test will compare the cuTile output against the oracle using the exact same input file:
- all required head dimensions;
- sequences equal to and larger than one tile;
- multiple divisible sequence lengths;
- multiple batch sizes and head counts;
- very small and large score magnitudes;
- invalid/non-divisible shapes rejected before launch;
- output NaN/Inf rejection.

The course proposal requires these categories.

## Regression policy

Every discovered correctness bug becomes a permanent regression test.

A performance change is accepted only after:
1. full correctness suite passes;
2. no NaN/Inf is introduced;
3. error metrics remain within the frozen tolerance;
4. invalid inputs are still rejected correctly.

## Test isolation

Tests of the numerical oracle must not depend on cuTile availability.

GPU tests may be skipped when the required GPU/software stack is absent during local development, but the A100 CI/release workflow will treat an unavailable required GPU environment as infrastructure failure rather than success.

## Reproducibility

All generated benchmark/test inputs use explicit seeds.

Benchmark data must record:
- git commit;
- GPU;
- driver;
- CUDA;
- cuTile;
- tile configuration;
- seed.
