# CUDA kernel versions

These are the ordinary CUDA C++ comparison kernels. They are not the main cuTile implementation.

| Version | Main change |
|---|---|
| v1 | One query row per block; shared K/V tile; two-pass softmax. |
| v2 | One query row per warp; warp-shuffle reductions and online softmax. |
| v3 | Computes four QK scores per loop iteration. |
| v4 | Keeps query fragments in registers across K tiles. |
| v5 | Uses half2 QK loads. |

The source files are kept under src/variants/ so earlier versions are not overwritten. The main shared-tile version is src/shared_tile_cuda_attention.py.

## How to compare versions

- Run the same correctness checks for every version.
- Keep the input shapes and tile settings fixed for a direct comparison.
- Use warmups, CUDA events and repeated samples.
- Report median time and variation.
- Call a version faster only when the repeated results support it.

Full timing history and negative results are in Report.md. Do not compare results from different GPUs as if they were measured under the same conditions.
