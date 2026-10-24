# cuTile kernel walkthrough

Source: src/cutile_attention.py  
Reference: src/reference.py  
Baseline: src/baseline.py

## What it computes

For each query, causal attention uses only the current token and earlier tokens:

O = softmax(QKᵀ / sqrt(D) + causal mask) V

A plain implementation stores the full N-by-N score and probability matrices. This kernel works on tiles instead.

## Kernel path

1. Validate shapes, dtype, layout and tile sizes before launch.
2. Load a query tile and initialise the running softmax state.
3. Walk through key/value tiles. Skip a tile if all its keys are in the future for the current query tile.
4. Compute QK scores, apply 1/sqrt(D), then mask future positions.
5. Update the running maximum, softmax sum and weighted-value accumulator.
6. Normalise the accumulated values and write the output tile.

For each query row, the running state is the maximum m, normalisation sum l and weighted value accumulator a. When a new score tile arrives, use m_new = max(m, m_tile), rescale the old l and a by exp(m - m_new), and add the new tile's contributions using exp(scores - m_new). The final output is a/l.

Q/K/V and output use FP16. Scores, softmax state and the final weighted-value accumulation use FP32. The kernel does not store the full score or probability matrix in global memory.

## Why the tests matter

The 34 executions in test_gpu_correctness.py compare supported workload/tile pairs with the independent reference and check stability, causal masking and the input contract. The 14 in test_gpu_edge_cases.py check known mathematical properties and invalid inputs. Together they make 48 GPU test executions.

The separate numerical matrix checks 29 legal workload/tile pairs. It is a different check from the 48-test suite.

## Comparison baseline

src/baseline.py is deliberately unfused FP32 CuPy attention. It creates the full score and probability matrices. A reported speedup is relative to that baseline only, not to FlashAttention or optimized PyTorch SDPA.

## Current evidence boundary

The last complete RTX 4060 run is #39422402438 on commit bf5a623. The kernel was compacted after that run, so repeat the core GPU checks on the current source before treating those numbers as current. The saved measurements and profiler analysis are linked from the [test matrix](TEST_MATRIX.md).
