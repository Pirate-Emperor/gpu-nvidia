# Native CUDA Variants

These implementations are deliberately independent of cuTile. They exist to exercise and validate the fused attention algorithm while the cuTile runtime is blocked by the node's current driver.

## Variant A: block-row fused CUDA

File: src/custom_cuda_attention.py

One CUDA block owns one query row. All 128 threads cooperatively reduce each Q·K dot product. A second pass performs stable softmax accumulation, with each active thread producing one output dimension.

Properties:
- FP16 input storage.
- FP32 dot-product/softmax/value accumulation.
- Two-pass max + normalization for numerical stability.
- No global score matrix.
- No global probability matrix.
- Uses block-wide __syncthreads().
- Supports D=32, 64, 128.

Primary diagnostic risks:
- synchronization errors,
- incorrect block reduction,
- incorrect broadcast of the running maximum,
- unnecessary synchronization/register pressure.

## Variant B: warp-row fused CUDA

File: src/warp_cuda_attention.py

One warp owns one query row. Four warps are packed into each CUDA block. Each lane contributes to the Q·K dot product, with __shfl_*_sync reductions. Each lane accumulates up to four output dimensions.

Properties:
- FP16 input storage.
- FP32 dot-product/softmax/value accumulation.
- Two-pass stable softmax.
- No global score/probability matrix.
- Warp-synchronous reduction instead of block-wide reduction.
- Supports D=32, 64, 128.

Primary diagnostic risks:
- warp reduction/broadcast mistakes,
- lane-to-output indexing mistakes,
- register pressure from multiple output accumulators.

## Cross-validation rule

Neither custom CUDA implementation is treated as its own oracle.

Every result must be checked against:
1. the independent full FP32 NumPy reference;
2. the independent tiled online-softmax reference;
3. the other custom CUDA implementation.

Only after these pass does a cuTile result become a useful comparison point.

## Relation to the course target

These are not substitutes for the required cuTile kernel. They are the low-level CUDA control implementations that let us debug the fused algorithm, precision, causal masking, synchronization, and indexing before the cuTile environment becomes available.

## Variant C: shared-memory tiled CUDA

File: src/shared_tile_cuda_attention.py

This variant stages K and V into explicit shared-memory tiles of 16, 32, or 64
keys. The current kernel handles four query rows per block (one warp per row),
stages a small query tile, and evaluates a stable online-softmax recurrence
tile by tile.

Properties:
- Explicit K/V shared-memory tiling.
- Tile sizes 32 and 64.
- FP16 storage with FP32 accumulation.
- No global score or probability matrix.
- Sequence length must be divisible by tile_k.
- Provides the closest ordinary-CUDA structural analogue to a tile DSL kernel.

Its purpose is especially important for the later cuTile work: if this version
and the online NumPy recurrence agree, most remaining debugging can be isolated
to cuTile syntax, indexing, launch/grid semantics, or compiler behavior.


## Preserved shared-tile optimization ladder

These are separate files so benchmark history is reproducible and earlier
implementations are not overwritten.

| Version | File | Main change | Evidence / decision |
|---|---|---|---|
| v1 | `src/variants/shared_tile_v1.py` | One query row/block; shared K/V; two-pass max then weighted accumulation. | Correct; deliberately simple performance baseline. |
| v2 | `src/variants/shared_tile_v2.py` | Four query rows/block, one warp per query row; warp-shuffle dot reduction and online softmax. | Correct; large measured improvement over v1 on medium/long cases. |
| v3 | `src/variants/shared_tile_v3.py` | Computes four QK scores per inner loop iteration. | Correct; often improves short/medium cases; long/batched performance varies. PTXAS used 40 registers. |
| Current | `src/shared_tile_cuda_attention.py` | Reuses each loaded Q fragment across the four score calculations. | Main A100/R550 surrogate. PTXAS: 32 registers, 33,792 B shared memory, no reported spills. |
| v4 | `src/variants/shared_tile_v4.py` | Hoists Q fragments into persistent registers across K tiles. | 30/30 isolated GPU tests passed, but timings mixed. PTXAS/cubin summary was identical to current, so not promoted. |
| v5 | `src/variants/shared_tile_v5.py` | Uses `half2` for adjacent FP16 Q/K values in dot-product calculation. | 30/30 isolated GPU tests passed; mixed timings and high variability mean the performance gain is not yet proven. |

All five are ordinary CUDA C++ compiled through CuPy RawKernel/NVCC for SM80.
They are structural surrogates for tile-oriented reasoning only; none is an
actual NVIDIA cuTile implementation. The version-ladder numbers and detailed
decisions are maintained in [the experiment report](EXPERIMENT_REPORT.md).

## Common GPU correctness gates

Dedicated A100 workflows check the tiled kernels against the independent FP32
reference across D=32/64/128, N=64/128/256, and tile_k=16/32/64, plus batched,
causal future-K/V invariance, long-sequence smoke, and finite-output checks.
The broad suite includes the current shared-tile path; v4/v5 have their own
30-test GPU workflow. A passing test does not by itself imply faster performance.
