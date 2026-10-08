# Kernel Version Ledger

| Version | Backend | Main change | Correctness status | Performance status |
|---|---|---|---|---|
| v1 | ordinary CUDA | 1 block/query row; shared K/V tiles; full-block reductions | Historical implementation | Pending benchmark |
| v2 | ordinary CUDA | 4 warps/block; 1 query row/warp; shared K/V tiles; one-pass online softmax | A100 validation pending | Pending benchmark |
| v3 | future | Only create after measured improvement over v2 | Not started | Not started |

## Comparison rule

1. A new version must pass the same correctness suite as its predecessor.
2. It must preserve the numerical, causal, and shape contracts.
3. Benchmark conditions must remain identical: resident inputs, warmups, CUDA events, and iteration count.
4. A version is called an improvement only after a measured steady-state median improvement on selected cases.

Never delete an older implementation after creating a new version.