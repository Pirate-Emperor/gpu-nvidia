# Baseline ladder

Each implementation answers a different question.

| Implementation | What it shows |
|---|---|
| FP32 NumPy reference | Mathematical result and tolerance check |
| Tiled online-softmax reference | Whether the streaming recurrence is correct |
| Unfused FP32 CuPy | GPU timing baseline with full score/probability intermediates |
| Scalar CUDA | A simple CUDA performance floor |
| Fused block-row CUDA | Benefit from fusing attention work |
| Fused warp-row CUDA | Effect of warp-level reductions |
| Shared-memory tiled CUDA | Effect of tiling and shared-memory reuse |
| NVIDIA cuTile | Main course implementation |

The CUDA C++ implementations are controls, not cuTile variants. The CuPy baseline is deliberately unfused; do not present its speedup as a win over FlashAttention or optimized PyTorch SDPA.

Only compare numbers measured on the same GPU with matching inputs and a controlled timing method. The full history is in Report.md.
