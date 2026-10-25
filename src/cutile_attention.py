"""cuTile implementation of fused causal scaled dot-product attention."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from contract import TileConfig, validate_launch_contract
from reference import AttentionValidationError


OUTPUT_DTYPE = np.float16


class CuTileUnavailableError(RuntimeError):
    """Raised when the cuTile/CuPy runtime is unavailable."""


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _import_runtime() -> tuple[Any, Any]:
    try:
        import cupy as cp
        import cuda.tile as ct
    except Exception as exc:  # pragma: no cover - depends on GPU environment
        raise CuTileUnavailableError(
            "cuTile/CuPy runtime is unavailable; use a GPU and driver/runtime "
            "supported by the installed cuTile package"
        ) from exc
    return cp, ct


@lru_cache(maxsize=64)
def _build_kernel(
    tile_q: int,
    tile_k: int,
    head_dim: int,
    value_dim: int,
    kv_groups: int = 1,
    load_latency: int = 0,
):
    # cuTile resolves kernel annotations (including ct.Constant[int]) using
    # the function's module globals. Keep imports lazy for CPU-only users, but
    # publish the runtime module globally before the decorator inspects them.
    global ct
    cp, ct = _import_runtime()

    @ct.kernel
    def kernel(q,k,v,o,tq:ct.Constant[int],tk:ct.Constant[int],d:ct.Constant[int],dv:ct.Constant[int],kg:ct.Constant[int],ll:ct.Constant[int],scl:ct.Constant[float]):
        b=ct.bid(0);h=ct.bid(1);qb=ct.bid(2)
        if ll==0:
            q4=ct.load(q,index=(b,h,qb,0),shape=(1,1,tq,d),padding_mode=ct.PaddingMode.ZERO)
        else:
            q4=ct.load(q,index=(b,h,qb,0),shape=(1,1,tq,d),padding_mode=ct.PaddingMode.ZERO,latency=ll)
        q0=ct.reshape(q4,(tq,d))
        m=ct.full((tq,1),float("-inf"),dtype=ct.float32)
        l=ct.zeros((tq,1),dtype=ct.float32)
        u=ct.zeros((tq,dv),dtype=ct.float32)
        qi=ct.reshape(ct.arange(tq,dtype=ct.int32),(tq,1))
        qi=qi+qb*tq
        nkt=ct.cdiv((qb+1)*tq,tk)
        kh=h//kg
        for kb in range(nkt):
            if ll==0:
                k4=ct.load(k,index=(b,kh,kb,0),shape=(1,1,tk,d),padding_mode=ct.PaddingMode.ZERO)
                v4=ct.load(v,index=(b,kh,kb,0),shape=(1,1,tk,dv),padding_mode=ct.PaddingMode.ZERO)
            else:
                k4=ct.load(k,index=(b,kh,kb,0),shape=(1,1,tk,d),padding_mode=ct.PaddingMode.ZERO,latency=ll)
                v4=ct.load(v,index=(b,kh,kb,0),shape=(1,1,tk,dv),padding_mode=ct.PaddingMode.ZERO,latency=ll)
            k0=ct.reshape(k4,(tk,d))
            v0=ct.reshape(v4,(tk,dv))
            kt=ct.transpose(k0)
            acc=ct.zeros((tq,tk),dtype=ct.float32)
            s=ct.mma(q0,kt,acc)
            s=s*scl
            ki=ct.reshape(ct.arange(tk,dtype=ct.int32),(1,tk))
            ki=ki+kb*tk
            fut=ki>qi
            s=ct.where(fut,float("-inf"),s)
            mt=ct.max(s,axis=1,keepdims=True)
            mn=ct.maximum(m,mt)
            a=ct.exp(m-mn)
            p=ct.exp(s-mn)
            ln=a*l+ct.sum(p,axis=1,keepdims=True)
            p16=p.astype(ct.float16)
            u0=a*u
            u=ct.mma(p16,v0,u0)
            l=ln
            m=mn
        out=u/l
        out4=ct.reshape(out,(1,1,tq,dv))
        ct.store(o,index=(b,h,qb,0),tile=out4.astype(o.dtype))

    return kernel


def _validate_cutile_config(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    tile_q: int,
    tile_k: int,
) -> TileConfig:
    tile = TileConfig(query=tile_q, key=tile_k)
    validate_launch_contract(q, k, v, tile)

    head_dim = q.shape[-1]
    value_dim = v.shape[-1]

    if head_dim not in (32, 64, 128):
        raise AttentionValidationError(
            f"head dimension D={head_dim} is unsupported; expected one of 32, 64, 128"
        )
    if value_dim != head_dim:
        raise AttentionValidationError(
            f"self-attention requires Dv == D; got D={head_dim}, Dv={value_dim}"
        )

    for name, value in (
        ("query tile", tile_q),
        ("key tile", tile_k),
        ("head dimension", head_dim),
        ("value dimension", value_dim),
    ):
        if not _is_power_of_two(value):
            raise AttentionValidationError(
                f"{name} must be a power of two for cuTile"
            )

    return tile


def launch_device(
    q_gpu,
    k_gpu,
    v_gpu,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
    load_latency: int = 0,
):
    """Launch fused causal attention on already resident GPU arrays.

    load_latency=0 lets the compiler infer load scheduling; values 1..10 are
    optional cuTile DRAM-traffic hints. On-device benchmarking must determine
    whether any hint improves this specific kernel.
    """
    cp, ct = _import_runtime()

    batch, heads, seq_len, head_dim = q_gpu.shape
    kv_heads = k_gpu.shape[1]
    value_dim = v_gpu.shape[-1]
    if q_gpu.ndim != 4 or k_gpu.ndim != 4 or v_gpu.ndim != 4:
        raise AttentionValidationError("Q/K/V device arrays must have rank 4")
    if k_gpu.shape[0] != batch or v_gpu.shape[0] != batch:
        raise AttentionValidationError("Q/K/V batch dimensions must match")
    if k_gpu.shape[1] != v_gpu.shape[1] or kv_heads < 1:
        raise AttentionValidationError("K and V must have the same positive KV-head count")
    if k_gpu.shape[2] != seq_len or v_gpu.shape[2] != seq_len:
        raise AttentionValidationError("Q/K/V sequence lengths must match")
    if k_gpu.shape[-1] != head_dim:
        raise AttentionValidationError("Q and K head dimensions must match")
    if heads % kv_heads != 0:
        raise AttentionValidationError("query-head count must be divisible by KV-head count for GQA/MQA")
    if load_latency != 0 and not 1 <= load_latency <= 10:
        raise AttentionValidationError("load_latency must be 0 (compiler default) or an integer from 1 to 10")
    kv_groups = heads // kv_heads
    if head_dim not in (32, 64, 128):
        raise AttentionValidationError(
            f"head dimension D={head_dim} is unsupported; expected one of 32, 64, 128"
        )
    if value_dim != head_dim:
        raise AttentionValidationError(
            f"self-attention requires Dv == D; got D={head_dim}, Dv={value_dim}"
        )

    tile = TileConfig(query=tile_q, key=tile_k)
    if seq_len % tile.query != 0 or seq_len % tile.key != 0:
        raise AttentionValidationError(
            f"sequence length {seq_len} must be divisible by query/key tiles "
            f"{tile.query}/{tile.key}"
        )
    for name, value in (
        ("query tile", tile_q),
        ("key tile", tile_k),
        ("head dimension", head_dim),
        ("value dimension", value_dim),
    ):
        if not _is_power_of_two(value):
            raise AttentionValidationError(
                f"{name} must be a power of two for cuTile"
            )

    out_gpu = cp.empty(
        (batch, heads, seq_len, value_dim),
        dtype=cp.float16,
    )

    kernel = _build_kernel(tile.query, tile.key, head_dim, value_dim, kv_groups, load_latency)
    grid = (
        batch,
        heads,
        seq_len // tile.query,
    )
    scale = 1.0 / np.sqrt(head_dim)

    kernel_args = (
        q_gpu,
        k_gpu,
        v_gpu,
        out_gpu,
        tile.query,
        tile.key,
        head_dim,
        value_dim,
        kv_groups,
        load_latency,
        float(scale),
    )

    ct_stream = cp.cuda.get_current_stream()
    ct.launch(ct_stream, grid, kernel, kernel_args)
    return out_gpu


def launch(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
    load_latency: int = 0,
):
    """Copy inputs to the GPU and launch fused causal attention."""
    cp, _ = _import_runtime()

    tile = _validate_cutile_config(q, k, v, tile_q, tile_k)
    q_gpu = cp.asarray(q)
    k_gpu = cp.asarray(k)
    v_gpu = cp.asarray(v)
    return launch_device(
        q_gpu,
        k_gpu,
        v_gpu,
        tile_q=tile.query,
        tile_k=tile.key,
        load_latency=load_latency,
    )


def forward_gqa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
) -> np.ndarray:
    """Optional grouped-query/multi-query causal attention.

    Q has Hq heads while K/V share Hkv heads, where Hq is divisible by Hkv.
    Each group of Hq/Hkv query heads reuses one K/V head. The core forward()
    contract remains the stricter equal-head self-attention path.
    """
    for name, x in (("Q", q), ("K", k), ("V", v)):
        if not isinstance(x, np.ndarray) or x.ndim != 4:
            raise AttentionValidationError(f"{name} must be a rank-4 NumPy array")
        if x.dtype != np.float16:
            raise AttentionValidationError(f"{name} must have dtype float16")
        if not x.flags.c_contiguous:
            raise AttentionValidationError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(x)):
            raise AttentionValidationError(f"{name} contains NaN or infinity")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise AttentionValidationError("Q/K/V batch dimensions must match")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise AttentionValidationError("Q/K/V sequence lengths must match")
    if k.shape[1] != v.shape[1] or k.shape[1] < 1:
        raise AttentionValidationError("K and V must have the same positive KV-head count")
    if q.shape[1] % k.shape[1] != 0:
        raise AttentionValidationError("query-head count must be divisible by KV-head count")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise AttentionValidationError("Q/K/V head dimensions must match")
    if q.shape[-1] not in (32, 64, 128):
        raise AttentionValidationError("head dimension must be one of 32, 64, 128")
    if q.shape[-2] % tile_q or q.shape[-2] % tile_k:
        raise AttentionValidationError(
            f"sequence length must be divisible by query/key tiles {tile_q}/{tile_k}"
        )
    for name, value in (("query tile", tile_q), ("key tile", tile_k)):
        if value <= 0 or (value & (value - 1)):
            raise AttentionValidationError(f"{name} must be a positive power of two")

    cp, _ = _import_runtime()
    out_gpu = launch_device(
        cp.asarray(q), cp.asarray(k), cp.asarray(v), tile_q=tile_q, tile_k=tile_k
    )
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)


def forward_packed_varlen(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    cu_seqlens,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
) -> np.ndarray:
    """Optional packed variable-length wrapper for independent causal sequences.

    Q/K/V use [1,H,total_tokens,D]; cu_seqlens is a monotone offset vector
    starting at zero and ending at total_tokens. Each segment is processed by
    an independent cuTile launch and copied into the packed output. This
    establishes packed-sequence semantics; it is deliberately not advertised
    as a one-launch fused varlen kernel.
    """
    offsets = np.asarray(cu_seqlens)
    if offsets.ndim != 1 or offsets.size < 2 or offsets.dtype.kind not in "iu":
        raise AttentionValidationError("cu_seqlens must be a 1-D integer offset array of length >= 2")
    if int(offsets[0]) != 0 or int(offsets[-1]) != q.shape[-2]:
        raise AttentionValidationError("cu_seqlens must start at 0 and end at total token count")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise AttentionValidationError("cu_seqlens offsets must be strictly increasing")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise AttentionValidationError("packed varlen currently expects batch dimension 1")
    if q.shape[-2] != k.shape[-2] or q.shape[-2] != v.shape[-2]:
        raise AttentionValidationError("packed Q/K/V must have the same total token count")
    outputs = []
    for start_raw, end_raw in zip(offsets[:-1], offsets[1:]):
        start, end = int(start_raw), int(end_raw)
        if (end - start) % tile_q or (end - start) % tile_k:
            raise AttentionValidationError(
                f"packed segment length {end-start} must be divisible by tiles {tile_q}/{tile_k}"
            )
        q_seg = np.ascontiguousarray(q[..., start:end, :])
        k_seg = np.ascontiguousarray(k[..., start:end, :])
        v_seg = np.ascontiguousarray(v[..., start:end, :])
        outputs.append(forward(q_seg, k_seg, v_seg, tile_q=tile_q, tile_k=tile_k))
    return np.concatenate(outputs, axis=-2)

def forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    tile_q: int = 64,
    tile_k: int = 64,
    load_latency: int = 0,
) -> np.ndarray:
    """Compute fused causal attention and return a NumPy FP16 output."""
    cp = _import_runtime()[0]
    out_gpu = launch(q, k, v, tile_q=tile_q, tile_k=tile_k, load_latency=load_latency)
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(out_gpu)
