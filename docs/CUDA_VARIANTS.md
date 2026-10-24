# Ordinary CUDA variants

These kernels are CUDA C++ implementations, not cuTile. They are kept as controls for checking the attention math and comparing low-level approaches.

## Implementations

| Variant | Source | Work assignment |
|---|---|---|
| Block-row | src/custom_cuda_attention.py | One CUDA block owns a query row; threads reduce Q·K scores and accumulate V. |
| Warp-row | src/warp_cuda_attention.py | One warp owns a query row; shuffle reductions handle Q·K. |
| Shared-tile | src/shared_tile_cuda_attention.py | A block owns a small query tile and stages K/V in shared memory. |

All use FP16 input storage and FP32 accumulation for scores, softmax and V. They avoid writing the full score/probability matrices to global memory and support D = 32, 64 and 128.

## What to check

- Block-row: reductions, synchronization and broadcast of the running maximum.
- Warp-row: shuffle reductions, lane indexing and register pressure.
- Shared-tile: shared-memory indexing, tile boundaries and occupancy.

## Cross-checks

Do not let an implementation validate itself. Compare each output with both:
- src/reference.py, the independent full-matrix FP32 NumPy reference;
- src/online_reference.py, the tiled online-softmax reference.

Cross-check the CUDA variants with each other too. Run correctness tests before comparing timing.

## Shared-tile history

The variants in src/variants/ keep the optimization steps separate:

| Version | Change |
|---|---|
| v1 | One query row per block; two-pass max and normalisation |
| v2 | Four query rows per block; one warp per row |
| v3 | Compute four QK scores per inner iteration |
| v4 | Keep query fragments in registers |
| v5 | Use half2 QK loads |

Historical results are recorded in Report.md. Do not call a result from these files a cuTile result.
