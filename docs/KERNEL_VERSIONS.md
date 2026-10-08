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

## v3 promotion benchmark

Method: 10 warmups, 30 CUDA-event timed iterations, resident inputs. Values are v2 median / v3 median; >1 means v3 is faster.

| Case | tile16 | tile32 | tile64 |
|---|---:|---:|---:|
| short_d64 | 2.03x | 1.20x | 0.91x |
| medium_d64 | 1.20x | 1.26x | 1.24x |
| medium_d128 | 1.14x | 1.15x | 1.15x |
| long_d128 | 1.07x | 1.03x | 1.01x |
| longer_batched_d128 | 1.09x | 0.97x | 0.90x |

v3/tile16 is the selected main configuration because it beat v2/tile16 on all five tested workloads and was the fastest v3 tile choice on every workload.
