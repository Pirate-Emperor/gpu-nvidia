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

This variant stages K and V into explicit shared-memory tiles of 32 or 64 keys.
The query row is staged once in shared memory, and the same stable two-pass
online-softmax recurrence is evaluated tile by tile.

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
