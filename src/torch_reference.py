"""Optional PyTorch SDPA backend for performance/reference comparison."""

from __future__ import annotations

import numpy as np

from reference import validate_attention_inputs


class TorchUnavailableError(RuntimeError):
    """Raised when PyTorch is unavailable."""


def _import_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - optional dependency
        raise TorchUnavailableError(
            "PyTorch is not installed; install a CUDA-enabled PyTorch build"
        ) from exc
    return torch, F


def forward(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Run causal SDPA with FP32 tensors and return FP16 NumPy output."""
    torch, F = _import_torch()
    validate_attention_inputs(q, k, v)

    if not torch.cuda.is_available():
        raise TorchUnavailableError("PyTorch CUDA runtime is unavailable")

    device = torch.device("cuda")
    q_t = torch.from_numpy(q).to(device=device, dtype=torch.float32)
    k_t = torch.from_numpy(k).to(device=device, dtype=torch.float32)
    v_t = torch.from_numpy(v).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

    return out.to(dtype=torch.float16).cpu().numpy()


def launch_device(q_t, k_t, v_t):
    """Run SDPA on resident CUDA tensors and return an FP16 tensor."""
    torch, F = _import_torch()
    if not torch.cuda.is_available():
        raise TorchUnavailableError("PyTorch CUDA runtime is unavailable")

    with torch.inference_mode():
        return F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        ).to(dtype=torch.float16)
