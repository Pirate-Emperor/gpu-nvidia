"""PyTorch bridge: cuTile forward, reference-only FP32 backward."""
from __future__ import annotations

import math

import torch


def _validate_torch_inputs(q, k, v):
    import torch

    for name, x in (("Q", q), ("K", k), ("V", v)):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError(f"{name} must be a rank-4 torch.Tensor")
        if x.dtype != torch.float16:
            raise ValueError(f"{name} must have dtype torch.float16")
        if not x.is_cuda:
            raise ValueError(f"{name} must be on CUDA")
        if not x.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if not torch.isfinite(x).all().item():
            raise ValueError(f"{name} contains NaN or infinity")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q/K/V shapes must match for the core self-attention wrapper")
    if q.shape[-1] not in (32, 64, 128):
        raise ValueError("head dimension must be one of 32, 64, 128")


class _CuTileCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, tile_q, tile_k):
        import cupy as cp
        import torch
        from cutile_attention import launch_device

        _validate_torch_inputs(q, k, v)
        # This custom autograd.Function owns gradient handling. DLPack exports
        # share the input storage with CuPy, but PyTorch rejects direct exports
        # of requires-grad tensors; detach the views while saving original q/k/v
        # below so backward still returns gradients for the original inputs.
        q_cp = cp.from_dlpack(q.detach())
        k_cp = cp.from_dlpack(k.detach())
        v_cp = cp.from_dlpack(v.detach())
        out_cp = launch_device(
            q_cp, k_cp, v_cp, tile_q=int(tile_q), tile_k=int(tile_k)
        )
        # Make the producer-stream completion explicit for this correctness-first
        # bridge; end-to-end timings include this interoperability cost.
        cp.cuda.get_current_stream().synchronize()
        out = torch.from_dlpack(out_cp)
        ctx.save_for_backward(q, k, v)
        ctx.tile_q = int(tile_q)
        ctx.tile_k = int(tile_k)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        import torch

        q, k, v = ctx.saved_tensors
        batch_heads = q.shape[0] * q.shape[1]
        n, d = q.shape[-2], q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        # Flatten batch and head into one explicit GEMM batch. Making every
        # transposed operand contiguous avoids the singleton-head cuBLAS SGEMM
        # INVALID_VALUE observed on Windows for torch.matmul([1,1,N,D], ...).
        # This remains a correctness-first FP32 reference backward.
        q32 = q.float().contiguous().reshape(batch_heads, n, d)
        k32 = k.float().contiguous().reshape(batch_heads, n, d)
        v32 = v.float().contiguous().reshape(batch_heads, n, d)
        grad = grad_output.float().contiguous().reshape(batch_heads, n, d)

        scores = torch.bmm(q32, k32.transpose(1, 2).contiguous()) * scale
        future = torch.triu(
            torch.ones((n, n), dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(future, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)

        grad_v = torch.bmm(probabilities.transpose(1, 2).contiguous(), grad)
        grad_p = torch.bmm(grad, v32.transpose(1, 2).contiguous())
        grad_s = probabilities * (
            grad_p - (grad_p * probabilities).sum(dim=-1, keepdim=True)
        )
        grad_s = grad_s.masked_fill(future, 0.0)
        grad_q = torch.bmm(grad_s, k32) * scale
        grad_k = torch.bmm(grad_s.transpose(1, 2).contiguous(), q32) * scale

        # Cast gradients back to the input dtype. This is an unfused FP32
        # reference backward and is not claimed as a performance optimization.
        return (
            grad_q.reshape_as(q).to(q.dtype),
            grad_k.reshape_as(k).to(k.dtype),
            grad_v.reshape_as(v).to(v.dtype),
            None,
            None,
        )


def cutile_attention_torch(q, k, v, *, tile_q=64, tile_k=64):
    """Call fused cuTile forward with PyTorch autograd-compatible output."""
    return _CuTileCausalAttention.apply(q, k, v, int(tile_q), int(tile_k))


def make_attention_module(torch=None):
    """Return the example causal self-attention module class lazily."""
    if torch is None:
        import torch

    import torch.nn as nn
    import torch.nn.functional as F

    class CuTileCausalSelfAttention(nn.Module):
        """Simple QKV-projection block using the cuTile forward bridge."""

        def __init__(self, embed_dim, num_heads, *, tile_q=64, tile_k=64, bias=True):
            super().__init__()
            if embed_dim % num_heads:
                raise ValueError("embed_dim must be divisible by num_heads")
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.head_dim = self.embed_dim // self.num_heads
            if self.head_dim not in (32, 64, 128):
                raise ValueError("per-head dimension must be one of 32, 64, 128")
            self.tile_q = int(tile_q)
            self.tile_k = int(tile_k)
            self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
            self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        def forward(self, x):
            if x.ndim != 3 or not x.is_cuda:
                raise ValueError("input x must be a CUDA tensor with shape [B,N,embed_dim]")
            batch, seq, _ = x.shape
            qkv = self.qkv(x).reshape(
                batch, seq, 3, self.num_heads, self.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            q = q.permute(0, 2, 1, 3).contiguous()
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
            from torch_integration import cutile_attention_torch
            y = cutile_attention_torch(
                q, k, v, tile_q=self.tile_q, tile_k=self.tile_k
            )
            y = y.transpose(1, 2).reshape(batch, seq, self.embed_dim)
            return self.proj(y)

    class SDPACausalSelfAttention(nn.Module):
        """Reference block with the same parameter layout as the cuTile block."""

        def __init__(self, embed_dim, num_heads, *, bias=True):
            super().__init__()
            if embed_dim % num_heads:
                raise ValueError("embed_dim must be divisible by num_heads")
            self.embed_dim = int(embed_dim)
            self.num_heads = int(num_heads)
            self.head_dim = self.embed_dim // self.num_heads
            self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
            self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        def forward(self, x):
            if x.ndim != 3 or not x.is_cuda:
                raise ValueError("input x must be a CUDA tensor with shape [B,N,embed_dim]")
            batch, seq, _ = x.shape
            qkv = self.qkv(x).reshape(
                batch, seq, 3, self.num_heads, self.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            q = q.permute(0, 2, 1, 3).contiguous()
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
            y = y.transpose(1, 2).reshape(batch, seq, self.embed_dim)
            return self.proj(y)

    return CuTileCausalSelfAttention, SDPACausalSelfAttention
