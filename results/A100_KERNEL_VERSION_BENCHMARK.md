# A100 Kernel Version Benchmark

Environment: NVIDIA A100-SXM4-40GB, SM80, driver R550.90.07, CUDA 12.4.
Method: 10 warmups, 30 CUDA-event timed iterations, resident GPU inputs, median latency.

## v1 -> v2
| Case | tile_k | v1 ms | v2 ms | Speedup |
|---|---:|---:|---:|---:|
| short_d32 | 32 | 0.105 | 0.060 | 1.75x |
| short_d32 | 64 | 0.105 | 0.060 | 1.75x |
| short_d64 | 32 | 0.114 | 0.048 | 2.37x |
| short_d64 | 64 | 0.100 | 0.068 | 1.48x |
| short_multihead_d64 | 32 | 0.354 | 0.078 | 4.55x |
| short_multihead_d64 | 64 | 0.364 | 0.078 | 4.67x |
| medium_d64 | 32 | 1.732 | 0.245 | 7.08x |
| medium_d64 | 64 | 1.725 | 0.244 | 7.08x |
| medium_d128 | 32 | 10.289 | 0.991 | 10.38x |
| medium_d128 | 64 | 10.245 | 0.986 | 10.39x |
| long_d128 | 32 | 94.486 | 6.790 | 13.92x |
| long_d128 | 64 | 87.835 | 7.353 | 11.94x |
| longer_batched_d128 | 32 | 410.587 | 30.544 | 13.44x |
| longer_batched_d128 | 64 | 385.523 | 30.246 | 12.75x |

## v2 -> v3
| Case | tile16 | tile32 | tile64 |
|---|---:|---:|---:|
| short_d64 | 2.03x | 1.20x | 0.91x |
| medium_d64 | 1.20x | 1.26x | 1.24x |
| medium_d128 | 1.14x | 1.15x | 1.15x |
| long_d128 | 1.07x | 1.03x | 1.01x |
| longer_batched_d128 | 1.09x | 0.97x | 0.90x |

v3/tile16 is the selected configuration. It is faster than the best v2 configuration on each of these five workloads.
