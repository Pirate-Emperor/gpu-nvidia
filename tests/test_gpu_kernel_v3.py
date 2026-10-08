"""Correctness regression for v2 and experimental v3 shared-tile kernels."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.gpu
if os.environ.get("REQUIRE_A100") == "1":
    try:
        import cupy as cp
    except ImportError as exc:
        pytest.fail(f"CuPy is required: {exc}", pytrace=False)
else:
    cp = pytest.importorskip("cupy")

from reference import causal_attention_fp32, passes_mixed_tolerance
from variants.shared_tile_v2 import launch_device as v2_launch
from variants.shared_tile_v3 import launch_device as v3_launch

def make_inputs(B,H,N,D,seed):
    rng=np.random.default_rng(seed)
    q=rng.standard_normal((B,H,N,D),dtype=np.float32).astype(np.float16)
    k=rng.standard_normal((B,H,N,D),dtype=np.float32).astype(np.float16)
    v=rng.standard_normal((B,H,N,D),dtype=np.float32).astype(np.float16)
    return q,k,v

@pytest.mark.parametrize("B,H,N,D,tile_k",[
    (1,1,64,32,32),
    (1,2,128,64,64),
    (1,2,256,128,32),
    (2,2,128,64,64),
    (2,4,256,128,64),
])
def test_v3_matches_v2_and_reference(B,H,N,D,tile_k):
    q,k,v=make_inputs(B,H,N,D,23000+B+H+N+D+tile_k)
    ref=causal_attention_fp32(q,k,v,output_dtype=np.float16)
    qg,kg,vg=cp.asarray(q),cp.asarray(k),cp.asarray(v)
    v2=cp.asnumpy(v2_launch(qg,kg,vg,tile_k=tile_k))
    v3=cp.asnumpy(v3_launch(qg,kg,vg,tile_k=tile_k))
    assert passes_mixed_tolerance(v2,ref,abs_scale=2e-2,rel_scale=2e-2)
    assert passes_mixed_tolerance(v3,ref,abs_scale=2e-2,rel_scale=2e-2)
    np.testing.assert_allclose(v3,v2,rtol=2e-2,atol=2e-2)

@pytest.mark.parametrize("tile_k",[16,32,64])
def test_v3_adversarial(tile_k):
    N=D=128
    q=np.zeros((1,1,N,D),dtype=np.float16)
    k=np.zeros_like(q)
    for i in range(N):
        q[...,i,i]=np.float16(64.0)
        k[...,i,i]=np.float16(64.0)
    v=np.zeros_like(q)
    v[..., :, 0]=np.arange(N,dtype=np.float16)
    qg,kg,vg=cp.asarray(q),cp.asarray(k),cp.asarray(v)
    out=cp.asnumpy(v3_launch(qg,kg,vg,tile_k=tile_k))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[0,0,:,0],np.arange(N,dtype=np.float16),rtol=2e-2,atol=2e-2)
