# Attention Baseline Ladder

The project keeps multiple implementations so mathematical correctness,
algorithmic recurrence, CUDA programming choices, and cuTile optimization can
be evaluated independently.

| Layer | Backend | Purpose | Course status |
|---|---|---|---|
| 0 | NumPy FP32 full attention | Independent mathematical oracle | Required reference |
| 1 | NumPy tiled online-softmax | Independent recurrence validation | Validation aid |
| 2 | CuPy unfused | Materialized score/probability GPU baseline | Required |
| 3 | Simple scalar CUDA | One-thread-per-output programming floor | Extension |
| 4 | Custom block-row CUDA fused | Ordinary CUDA fused implementation with cooperative dot-product reduction | Diagnostic/bridge |
| 5 | Custom warp-row CUDA fused | Warp-shuffle fused implementation | Diagnostic/bridge |
| 6 | PyTorch SDPA | Framework/vendor performance reference | Recommended |
| 7 | cuTile fused | Main project fused implementation | Required target |

## Diagnostic value

- If the mathematical reference and online recurrence disagree, the attention
  recurrence itself is wrong before any GPU work begins.
- If either custom CUDA fused implementation disagrees with layer 0, the problem
  can be debugged without involving cuTile.
- If the two independent custom CUDA kernels agree with each other and layer 0,
  the fused algorithm is strongly cross-validated before cuTile execution.
- If cuTile differs from the custom CUDA variants and layer 0, the issue is
  likely in the cuTile implementation or its tile/indexing semantics.
- If cuTile matches correctness but loses to the custom CUDA variants, profiling
  can focus on tiling, occupancy, memory traffic, and compiler mapping.
- Layer 1 independently validates the running-max/running-normalization
  recurrence used by both custom CUDA kernels and the cuTile target.

## Why the custom CUDA variants matter

They intentionally use **no cuTile**. They are not replacements for the
course deliverable. They are engineering bridges:

1. They allow fused attention to be executed and debugged on an A100 while the
   cuTile driver prerequisite is unavailable.
2. They provide an independent implementation against which the future cuTile
   kernel can be compared.
3. They expose CUDA synchronization, reduction, register-pressure, and warp
   mapping decisions that are directly useful when optimizing the cuTile
   version later.

## Course-scope boundary

The required course path is causal self-attention with FP16 inputs and FP32
accumulation, supported head dimensions, tile-divisible sequence lengths,
fused score/mask/softmax/value accumulation, at least two tile configurations,
correctness tests, rigorous timing, and profiler evidence.

BF16/FP8, backward propagation, grouped/multi-query attention, KV cache,
packed variable-length sequences, dropout, partial edge tiles, and alternative
DSL implementations remain optional/out of required scope until the required
path is complete.
