# Attention Baseline Ladder

The project keeps multiple implementations so correctness, algorithmic ideas, and GPU-programming performance can be separated.

| Layer | Backend | Purpose | Course status |
|---|---|---|---|
| 0 | NumPy FP32 full attention | Independent mathematical oracle | Required reference |
| 1 | NumPy tiled online-softmax | Independent recurrence validation | Validation aid |
| 2 | CuPy unfused | Materialized score/probability GPU baseline | Required |
| 3 | Simple scalar CUDA | Basic CUDA performance floor | Extension |
| 4 | PyTorch SDPA | Framework/vendor performance reference | Recommended |
| 5 | cuTile fused | Main fused implementation | Required target |

## Diagnostic value

- If cuTile differs from layer 0, the fused implementation is incorrect.
- If cuTile matches layer 0 but loses to layer 3, tile mapping/compiler/resource usage needs work.
- If cuTile beats layer 2 but loses to layer 4, the remaining gap should be profiled and explained.
- Layer 1 independently validates the running-max/running-normalization recurrence used by the fused kernel.

## Course-scope boundary

The required course path is causal self-attention with FP16 inputs and FP32 accumulation, supported head dimensions, tile-divisible sequence lengths, fused score/mask/softmax/value accumulation, at least two tile configurations, correctness tests, rigorous timing, and profiler evidence.

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache, packed variable-length sequences, dropout, partial edge tiles, and alternative DSL implementations remain optional/out of required scope until the required path is complete.
